import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from Tasks.localizaciones import generador_static_validación
from Tasks.localizaciones import generador_coordenadas_clientes
from Models.actor import DRL4TSP
from Tasks.vrp import VehicleRoutingDataset
from Tasks.vrp import reward
from Models.critc import StateCritic
from collections import Counter
#¿Qué dispositivo se va a utilizar?
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Debemos definir los parámetros a utilizar
def cargar_parametros(actor, critic, checkpoint_dir):
    #ACTOR
    ruta_actor = os.path.join(checkpoint_dir, 'actor.pt')
    actor.load_state_dict(torch.load(ruta_actor, map_location =device))
    #CRÍTICO
    critic_path = os.path.join(checkpoint_dir, 'critic.pt')
    critic.load_state_dict(torch.load(critic_path, map_location=device))

#Nos calcula la recompensa obtenida en el tour generado por el agente
def validate(data_loader, actor, reward):
    actor.eval()

    recompensas = []
    tour_indices_final = []
    for batch_idx, batch in enumerate(data_loader):
        static, dynamic, x0 = batch
        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        with torch.no_grad():
            tour_indices, _ = actor.forward(static, dynamic, x0)

        recompensa = reward(static, tour_indices).mean().item()
        recompensas.append(recompensa)
        tour_indices_final.append(tour_indices)
        #if render is not None and mejor_recompensa == recompensa_actual:
            #grafica_tours = render(static, tour_mejor_recompensa)
    return np.mean(recompensas), tour_indices_final

def main(args):
    LOAD_DICT = {10: 20, 20: 30, 50: 40, 100: 50}
    MAX_DEMAND = 9
    STATIC_SIZE = 2  # (x, y)
    DYNAMIC_SIZE = 2  # (load, demand)

    max_load = LOAD_DICT[args.num_nodes]

    #Generar datos de prueba
    localizacion_clientes_test = generador_coordenadas_clientes(args.num_nodes, STATIC_SIZE)
    static_location_test = generador_static_validación(args.num_samples,localizacion_clientes_test)
    test_data = VehicleRoutingDataset(args.num_samples,
                                      args.num_nodes,
                                      max_load,
                                      MAX_DEMAND,
                                      localizacion_clientes_test,
                                      static_location_test,
                                      STATIC_SIZE,
                                      args.seed)
    test_loader = DataLoader(test_data, args.batch_size, shuffle=False, num_workers=0)
    #Inicializar el modelo
    actor = DRL4TSP(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    args.hidden_size,
                    test_data.update_dynamic,
                    test_data.update_mask,
                    args.num_layers,
                    args.dropout).to(device)
    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size)

    #Directorio a cambiar
    checkpoint_dir = r'C:\Users\azurs\PycharmProjects\RL-VRP-PtrNtwrk\vrp\20\17_02_31.520123\checkpoints\49'
    cargar_parametros(actor, critic, checkpoint_dir)

    #Recompensa y tour obtenido en con el conjunto de test
    recompensa, tours_escogidos = validate(test_loader, actor, reward)
    #Estas líneas nos inducan cuál es el tour que más se ha repetido, y por tanto corresponde al mejor tour encontrado por el agente a lo largo de todas las instancias impuestas.
    tours = tours_escogidos[0]
    lista = tours.tolist()
    #Una vez cambiamos la forma de la variable tours a lista podemos obtener cuál es el tour más repetido y su frecuencia
    tuplas = [tuple(i) for i in lista]
    estadistica = Counter(tuplas)
    tour_mas_repetido, frecuencia = estadistica.most_common(1)[0]
    print(f'La distancia total del conjunto de test es de: {recompensa:.3f}\n'
          f'El tour más repetido por el agente, con un número total de {frecuencia} veces, \n'
          f'es: {tour_mas_repetido}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Combinatorial Optimization')
    parser.add_argument('--seed', default=12345, type=int)
    #parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--nodes', dest='num_nodes', default=20, type=int)
    parser.add_argument('--batch_size', default=150, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=150, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--num_samples', default=150, type=int)
    #En la etapa de test para escoger el mejor tour que se adapte a cambios de demanda se propone usar siempre el mismo número de instancias que de lotes

    args = parser.parse_args()

    main(args)
