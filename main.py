import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from Tasks.localizaciones import generador_static_entrenamiento
from Tasks.localizaciones import generador_static_validación
from Tasks.localizaciones import generador_coordenadas_clientes
from Models.actor import DRL4TSP
from Tasks import vrp
from Tasks.vrp import VehicleRoutingDataset
from Models.critc import StateCritic

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled=False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Detected device {}'.format(device))


def validate(data_loader, actor, reward_fn, render_fn=None, save_dir='.',
             num_plot=5):
    """Used to monitor progress on a validation set & optionally plot solution."""
    actor.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    for batch_idx, batch in enumerate(data_loader):
    #El data loader contiene una serie de lotes de datos que contienen batch_idx (indice de lotes) y batch(static, dynamic, x0)
        static, dynamic, x0 = batch

        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        with torch.no_grad():
            tour_indices, _ = actor.forward(static, dynamic, x0)

        reward = reward_fn(static, tour_indices).mean().item() #Utiliza la función recompensa definida en el script VRP
        rewards.append(reward)

        if render_fn is not None and batch_idx < num_plot:
            name = 'batch%d_%2.4f.png'%(batch_idx, reward)
            path = os.path.join(save_dir, name)
            render_fn(static, tour_indices, path)

    actor.train()
    return np.mean(rewards)

def train(actor, critic, task, num_nodes, train_data, valid_data, reward_fn,
          render_fn, batch_size, actor_lr, critic_lr, max_grad_norm,
          **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""
    ## En el artículo comentan que en el método de entrenamiento implementaron dos redes para el algoritmo de gradiente de políticas
    ## ACTOR NETWORKK: Predice la distribución de probabilidad a lo largo de la siguiente acción {actor_lr}
    ## CRITIC NET: Estima la recompensa para cualquier intancia del problema en cualquier estado del mismo. {critic_lr}
    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    save_dir = os.path.join(task, '%d' % num_nodes, now)

    print('Starting training')

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)  ## optim.Adam: implementa el algoritmo de Adam: Usado para mejorar el proceso de aprendizaje de un modelo
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)



    best_params = None
    best_reward = np.inf ##np.inf es una constante de la librería NumPy, esta constante representa un valor positivo e infinito == M en optimización lineal

    for epoch in range(20):

        actor.train()
        critic.train()

        ##2º cambio: en esta línea de código se está estableciendo las salidas del entrenamiento y como responde en cada iteración ("epoch"), en el momento que queramos moficar los outputs de entrenamiento y de test
        ##del modelo vamos a tener que modificar estos aspectos, deberemos haber creado unas variables adicionales en el script de critic, etc.
        times, losses, rewards, critic_rewards = [], [], [], []

        epoch_start = time.time()
        start = epoch_start

        for batch_idx, batch in enumerate(train_loader):

            static, dynamic, x0 = batch

            static = static.to(device)
            dynamic = dynamic.to(device)
            x0 = x0.to(device) if len(x0) > 0 else None

            # Full forward pass through the dataset
            tour_indices, tour_logp = actor(static, dynamic, x0)

            # Sum the log probabilities for each city in the tour
            reward = reward_fn(static, tour_indices)

            # Query the critic for an estimate of the reward
            critic_est = critic(static, dynamic).view(-1) ## el .view(-1) se utiliza para convertir el tensor de salida multidimensional en un tensor unidimensional (usado para conectar capas densas)

            advantage = (reward - critic_est)
            actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)

            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()

            critic_rewards.append(torch.mean(critic_est.detach()).item())
            rewards.append(torch.mean(reward.detach()).item())
            losses.append(torch.mean(actor_loss.detach()).item())

            if (batch_idx + 1) % 100 == 0: #### Cuando las instancias totales a resolver del conjunto de entrenamiento sean menor de batch_size * 100 no se inicializará este condicional
                end = time.time()
                times.append(end - start)
                start = end

                mean_loss = np.mean(losses[-100:])
                mean_reward = np.mean(rewards[-100:])

                print('  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs' %
                      (batch_idx, len(train_loader), mean_reward, mean_loss,
                       times[-1]))

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)

        # Save the weights
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)

        save_path = os.path.join(epoch_dir, 'critic.pt')
        torch.save(critic.state_dict(), save_path)

        # Save rendering of validation set tours
        valid_dir = os.path.join(save_dir, '%s' % epoch)

        mean_valid = validate(valid_loader, actor, reward_fn, render_fn,
                              valid_dir, num_plot=5)

        ###Posible cambio (tasa de mejora en cada epoch, excepto en la primera epoch (la tasa de mejora será siempre de un 100%)
        if epoch != 0:
            mejora_mejor_solucion = 100 * (best_reward - mean_valid) / best_reward # En cada epoch nos indica el porcentaje de mejora (cuando la mean_valid sea menor que best_reward tendremos un porcentaje de mejora positivo)
            mejora_conjunto_entrenamiento = 100 * (mean_reward - mean_valid) / mean_reward # En cada epoch nos indica el porcentaje de mejora frente al conjunto de entrenamietno
            print(f'Epoch: {epoch + 1}. La recompensa media obtenida en el conjunto de validación: {mean_valid: .3f}.\n'
                  f'La mejora obtenida en esta epoch frente a la mejor recompensa obtenida es del: {mejora_mejor_solucion: .2f}% \n'
                  f'La solución obtenida en el conjunto de validación mejora en un {mejora_conjunto_entrenamiento: .2f}% a la obtenida en el conjunto de entrenamiento')

        else:
            primera_solucion = mean_valid
            ##guardamos en una variable cuál ha sido el primer valor obtenido en la fase de entrenamiento para poder ver la mejora total del método
        # Save best model parameters
        if mean_valid < best_reward:

            best_reward = mean_valid

            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

            save_path = os.path.join(save_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)
        if args.train_size / batch_size < 100:
            print('Mean epoch loss/reward: %2.4f, %2.4f, %2.4f and took: %2.4fs\n' %
                  (mean_loss, mean_reward, mean_valid, time.time() - epoch_start))
        else:
            print('Mean epoch loss/reward: %2.4f, %2.4f, %2.4f, took: %2.4fs '\
                '(%2.4fs / 100 batches)\n' % \
                (mean_loss, mean_reward, mean_valid, time.time() - epoch_start,
                np.mean(times)))

    mejora_total = 100*(primera_solucion - best_reward) / primera_solucion
    print(f'La solución final obtenida ha mejorado en un {mejora_total:.2f}% a la primera ({primera_solucion: .2f})')


def train_vrp(args):

    # Goals from paper:
    # VRP10, Capacity 20:  4.84  (Greedy)
    # VRP20, Capacity 30:  6.59  (Greedy)
    # VRP50, Capacity 40:  11.39 (Greedy)
    # VRP100, Capacity 50: 17.23  (Greedy)

    print('Starting VRP training')
    #Lo primero que debemos hacer al entrenar el código es inicializar las localizaciones, este caso serán fijas
    # Determines the maximum amount of load for a vehicle based on num nodes
    LOAD_DICT = {10: 20, 20: 30, 50: 40, 100: 50}
    MAX_DEMAND = 9
    STATIC_SIZE = 2 # (x, y)
    DYNAMIC_SIZE = 2 # (load, demand)

    max_load = LOAD_DICT[args.num_nodes]

    #Lo primero que debemos hacer al entrenar el código es inicializar las localizaciones, este caso serán fijas
    localizacion_clientes_validacion = generador_coordenadas_clientes(args.num_nodes, STATIC_SIZE)
    localizacion_clientes_test = generador_coordenadas_clientes(args.num_nodes, STATIC_SIZE)
    static_locations_training = generador_static_entrenamiento(args.train_size, STATIC_SIZE, args.num_nodes)
    static_locations_validation = generador_static_validación(args.valid_size, localizacion_clientes_validacion)
    static_locations_test = generador_static_validación(args.valid_size, localizacion_clientes_test)
    #Cada vez que aparece la clase VehicleRoutingDataset tuve que añadir la entrada correspondiente a STATIC_SIZE y static_locations (tanto de entrenamiento como de validación)
    train_data = VehicleRoutingDataset(args.train_size,
                                       args.num_nodes,
                                       max_load,
                                       MAX_DEMAND,
                                       None,
                                       static_locations_training,
                                       STATIC_SIZE,
                                       args.seed)

    print('Train data: {}'.format(train_data))
    valid_data = VehicleRoutingDataset(args.valid_size,
                                       args.num_nodes,
                                       max_load,
                                       MAX_DEMAND,
                                       localizacion_clientes_validacion,
                                       static_locations_validation,
                                       STATIC_SIZE,
                                       args.seed + 1)

    actor = DRL4TSP(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    args.hidden_size,
                    train_data.update_dynamic,
                    train_data.update_mask,
                    args.num_layers,
                    args.dropout).to(device)
    print('Actor: {} '.format(actor))

    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)

    print('Critic: {}'.format(critic))

    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = vrp.reward
    kwargs['render_fn'] = vrp.render

    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))

    if not args.test:
        train(actor, critic, **kwargs)

    test_data = VehicleRoutingDataset(args.valid_size,
                                      args.num_nodes,
                                      max_load,
                                      MAX_DEMAND,
                                      localizacion_clientes_test,
                                      static_locations_test,
                                      STATIC_SIZE,
                                      args.seed + 2)

    test_dir = 'test'
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    out = validate(test_loader, actor, vrp.reward, vrp.render, test_dir, num_plot=5)

    print('Average tour length: ', out)
##Para entrenar el modelo deseado debemos usar num_nodes (tenemos un total de num_nodes localizaciones fijas, 1 correspondiente al depósito y otras num_nodes-1 correspodientes a nuestros clientes)
##Cabe destacar que el numsamples corresponde al número de muestras usadas en la base de datos.
##Muestreo: Se le conoce como muestreo a la técnica para la selección de una muestra a partir de una población estadística. Al elegir una muestra aleatoria
##se espera conseguir que sus propiedades sean extrapolables a la población. Este proceso permite obtener resultados parecidos a los que se alcanzarían si se realizase un estudio a toda la población.
##¿Sería necessario tener en cuenta el número de muestras si estamos suponiendo que tenemos unos clientes fijos con unas coordenadas fijas que son inalterables en el tiempo?
## Para el segundo cambio  "Cambiar los argumentos y la salida del modelo" Deberemos hacer algún cambio en los argumentos, por ejemplo, elegir el número de nodos.
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--task', default='vrp')
    parser.add_argument('--nodes', dest='num_nodes', default=20, type=int)
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size',default=1000, type=int) ## Entradas para ver el modelo, en la fase de mejora del código disminuimo el número de instancias del modelo de entrenamiento y validación
    parser.add_argument('--valid-size', default=10, type=int) ##20% del conjunto de entrenamiento

    args = parser.parse_args()

    #print('NOTE: SETTTING CHECKPOINT: ')
    #args.checkpoint = os.path.join('vrp', '10', '12_59_47.350165' + os.path.sep)
    #print(args.checkpoint)

    
    
    train_vrp(args)