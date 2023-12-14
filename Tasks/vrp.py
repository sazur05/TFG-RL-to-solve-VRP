"""Defines the main task for the VRP.

The VRP is defined by the following traits:
    1. Each city has a demand in [1, 9], which must be serviced by the vehicle
    2. Each vehicle has a capacity (depends on problem), the must visit all cities
    3. When the vehicle load is 0, it __must__ return to the depot to refill
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from Tasks.localizaciones import generador_static_entrenamiento
from Tasks.localizaciones import generador_static_validación
from Tasks.localizaciones import generador_coordenadas_clientes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

##Para el cambio de las posiciones fijas tuve que cambiar las variables de entrada de la clase, añadí las entradas STATIC_SIZE = 2 y static_locations = None
class VehicleRoutingDataset(Dataset):
    def __init__(self, num_samples, input_size, max_load=20, max_demand=9,
                 static_locations=None,
                 static_locations_final=None, STATIC_SIZE=2, seed=None):
        super(VehicleRoutingDataset, self).__init__()
        #super() se utiliza para llamar a un método usado en la clase padre, esta llamada se utiliza en las clases hijas a modo de herencia
        #1er cambio: 20 localizaciones fijas para los clientes de una empresa (empresa proveedora debe satisfacer las necesidades de sus clientes, fijos e inalterables)

        if max_load < max_demand:
            raise ValueError(':param max_load: must be > max_demand')

        if seed is None:
            seed = np.random.randint(1234567890)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # útil en situaciones en las que se desee reproducibilidad en experimentos o entrenamientos que involucren operaciones aleatorias.

        self.num_samples = num_samples
        self.max_load = max_load
        self.max_demand = max_demand
        ##Cambio aquí no había inicializado la variable STATIC_SIZE
        self.STATIC_SIZE = STATIC_SIZE
        #Añadí estas dos líneas de código para inicializar el tensor de las localizaciones antes del apartado de entrenamiento, para que realmente esas localizaciones sean fijas
        if static_locations is None:
            if num_samples == 'args.valid_size':
                static_locations = generador_coordenadas_clientes(input_size, STATIC_SIZE)
            elif num_samples == 'args.train_size':
                static_locations = None

        if static_locations_final is None:
            if num_samples == 'args.valid_size':
                static_locations_final = generador_static_validación(num_samples, static_locations)
            elif num_samples == 'args.train_size':
                static_locations_final = generador_static_entrenamiento(num_samples, STATIC_SIZE, input_size)

        #1er cambio: locations = torch.rand((num_samples, 2, input_size + 1)) -->
        self.static = static_locations_final

        # All states will broadcast the drivers current load
        # Note that we only use a load between [0, 1] to prevent large
        # numbers entering the neural network
        dynamic_shape = (num_samples, 1, input_size + 1)
        ##3er cambio: Demandas y cargas distintas
        ##Añadir otra variable. loads2 = torch.full(dynamic_shape, valor a usar)
        ##torch.full--> crea un tensor de dimensiones especificadas y con un valor de relleno torch.full(dim_size, fill_value)
        loads = torch.full(dynamic_shape, 1.)
        ##Concatenar ambas variables en un tensor final llamado cargas totales (un mismo vehículo puede afrontar dos tipos de pedidos distintos)


        # All states will have their own intrinsic demand in [1, max_demand), 
        # then scaled by the maximum load. E.g. if load=10 and max_demand=30, 
        # demands will be scaled to the range (0, 3)
        ##3er cambio: Añadir otra variable demands con el otro tipo de demandas y volver a concatenar ambos tensores
        demands = torch.randint(1, max_demand + 1, dynamic_shape)
        demands = demands / float(max_load)

        ##Hay que modificar este comando de manera que el depóstio tenga ambas demandas en el instante 0 sean nulas
        demands[:, 0, 0] = 0  # depot starts with a demand of 0
        self.dynamic = torch.tensor(np.concatenate((loads, demands), axis=1))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        ##print(f'El tensor static es del tipo: {self.static.type}')
        ##print(f'Salida del depóstio: {self.static[idx, :, 0:1]}')
        return (self.static[idx], self.dynamic[idx], self.static[idx, :, 0:1])

    ### Overrides method in DataSet significa que en este script se ha utilizado un conjunto de datos personalizado
    ## Específicamente con las dos definiciones anteriores (sacar la longitud de la base de datos y obtener un item de esta base según lo especificado de output)
    ## El último output de la función getitem extrae la primera columna de todas las filas del tensor static (¿Extrae el depósito?)

    def update_mask(self, mask, dynamic, chosen_idx=None):
        """Updates the mask used to hide non-valid states.

        Parameters
        ----------
        dynamic: torch.autograd.Variable of size (1, num_feats, seq_len)
        """

        # Convert floating point to integers for calculations
        loads = dynamic.data[:, 0]  # (batch_size, seq_len)
        demands = dynamic.data[:, 1]  # (batch_size, seq_len)

        # If there is no positive demand left, we can end the tour.
        # Note that the first node is the depot, which always has a negative demand
        if demands.eq(0).all():
            return demands * 0.

        # Otherwise, we can choose to go anywhere where demand is > 0
        new_mask = demands.ne(0) * demands.lt(loads)

        # We should avoid traveling to the depot back-to-back
        repeat_home = chosen_idx.ne(0) ## & choosen_idx.ne(1) (debemos asegurarnos que ninguno de los dos depositos es escogido)

        if repeat_home.any():
            new_mask[repeat_home.nonzero(), 0] = 1. #### new_mask[repeat_home.nonzero(), : 2] = 1. Actualiza las dos primeras columnas (son las que están referidas a ambos depósitos)
        #Sergio: Cambio del original 1- repeat_home--> ~repeat_home
        if (~repeat_home).any():
            new_mask[(~repeat_home).nonzero(), 0] = 0. ###new_mask[repeat_home.nonzero(), : 2] = 0.

        # ... unless we're waiting for all other samples in a minibatch to finish
        has_no_load = loads[:, 0].eq(0).float() ###has_no_load = loads[:, :2].eq(0).float()
        has_no_demand = demands[:, 1:].sum(1).eq(0).float() ###has_no_demand = demands[:, 2:].sum(1).eq(0).float()

        combined = (has_no_load + has_no_demand).gt(0)
        if combined.any():
            new_mask[combined.nonzero(), 0] = 1. ##new_mask[combined.nonzero(), :2] = 1.
            new_mask[combined.nonzero(), 1:] = 0. ##new_mask[combined.nonzero(), 2:] = 0.

        return new_mask.float()

    def update_dynamic(self, dynamic, chosen_idx):
        """Updates the (load, demand) dataset values."""

        # Update the dynamic elements differently for if we visit depot vs. a city
        visit = chosen_idx.ne(0)  # & visit = chosen_idx.ne(1)
        depot = chosen_idx.eq(0) #visit = chosen_idx.eq(1)

        # Clone the dynamic variable so we don't mess up graph
        all_loads = dynamic[:, 0].clone()
        all_demands = dynamic[:, 1].clone()

        load = torch.gather(all_loads, 1, chosen_idx.unsqueeze(1))
        demand = torch.gather(all_demands, 1, chosen_idx.unsqueeze(1))

        # Across the minibatch - if we've chosen to visit a city, try to satisfy
        # as much demand as possible
        if visit.any():

            new_load = torch.clamp(load - demand, min=0)
            new_demand = torch.clamp(demand - load, min=0)

            # Broadcast the load to all nodes, but update demand separately

            visit_idx = visit.nonzero().squeeze()

            all_loads[visit_idx] = new_load[visit_idx]
            all_demands[visit_idx, chosen_idx[visit_idx]] = new_demand[visit_idx].view(-1)
            all_demands[visit_idx, 0] = -1. + new_load[visit_idx].view(-1) #all_demands[visit_idx, :2
            # ] = -1. + new_load[visit_idx].view(-1)

        # Return to depot to fill vehicle load
        if depot.any():
            all_loads[depot.nonzero().squeeze()] = 1.
            all_demands[depot.nonzero().squeeze(), 0] = 0.

        tensor = torch.cat((all_loads.unsqueeze(1), all_demands.unsqueeze(1)), 1)
        return torch.tensor(tensor.data, device=dynamic.device)


def reward(static, tour_indices):
    """
    Euclidean distance between all cities / nodes given by tour_indices
    """

    # Convert the indices back into a tour
    idx = tour_indices.unsqueeze(1).expand(-1, static.size(1), -1)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)

    # Ensure we're always returning to the depot - note the extra concat
    # won't add any extra loss, as the euclidean distance between consecutive
    # points is 0
    start = static.data[:, :, 0].unsqueeze(1) ## Siempre empezamos desde el primer depóstio el ubicado en el origen de coordenadas
    y = torch.cat((start, tour, start), dim=1)

    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))

    return tour_len.sum(1)
##La recompensa que recibe el agente es directamente proporcional a la distancia total del tour, por tanto el gente va a recibir como mejor recompensa la mínima obtenida


def render(static, tour_indices, save_path):
    """Plots the found solution."""

    plt.close('all')

    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        idx = idx.expand(static.size(1), -1)
        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        start = static[i, :, 0].cpu().data.numpy()
        x = np.hstack((start[0], data[0], start[0]))
        y = np.hstack((start[1], data[1], start[1]))

        # Assign each subtour a different colour & label in order traveled
        idx = np.hstack((0, tour_indices[i].cpu().numpy().flatten(), 0))
        where = np.where(idx == 0)[0]

        for j in range(len(where) - 1):

            low = where[j]
            high = where[j + 1]

            if low + 1 == high:
                continue

            ax.plot(x[low: high + 1], y[low: high + 1], zorder=1, label=j)

        ax.legend(loc="upper right", fontsize=3, framealpha=0.5)
        ax.scatter(x, y, s=4, c='r', zorder=2)
        ax.scatter(x[0], y[0], s=20, c='k', marker='*', zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)


'''
def render(static, tour_indices, save_path):
    """Plots the found solution."""

    path = 'C:/Users/Matt/Documents/ffmpeg-3.4.2-win64-static/bin/ffmpeg.exe'
    plt.rcParams['animation.ffmpeg_path'] = path

    plt.close('all')

    num_plots = min(int(np.sqrt(len(tour_indices))), 3)
    fig, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                             sharex='col', sharey='row')
    axes = [a for ax in axes for a in ax]

    all_lines = []
    all_tours = []
    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        idx = idx.expand(static.size(1), -1)
        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        start = static[i, :, 0].cpu().data.numpy()
        x = np.hstack((start[0], data[0], start[0]))
        y = np.hstack((start[1], data[1], start[1]))

        cur_tour = np.vstack((x, y))

        all_tours.append(cur_tour)
        all_lines.append(ax.plot([], [])[0])

        ax.scatter(x, y, s=4, c='r', zorder=2)
        ax.scatter(x[0], y[0], s=20, c='k', marker='*', zorder=3)

    from matplotlib.animation import FuncAnimation

    tours = all_tours

    def update(idx):

        for i, line in enumerate(all_lines):

            if idx >= tours[i].shape[1]:
                continue

            data = tours[i][:, idx]

            xy_data = line.get_xydata()
            xy_data = np.vstack((xy_data, np.atleast_2d(data)))

            line.set_data(xy_data[:, 0], xy_data[:, 1])
            line.set_linewidth(0.75)

        return all_lines

    anim = FuncAnimation(fig, update, init_func=None,
                         frames=100, interval=200, blit=False,
                         repeat=False)

    anim.save('line.mp4', dpi=160)
    plt.show()

    import sys
    sys.exit(1)
'''
