#Cambios hechos y posibilades:
#1) Añadir localizaciones fijas
#2) Cambiar el front end del usuario (que indicadores de rendimiento) ---> Se podría añadir variables dependientes del recorrido y tiempo y así observar el gasto por parte de la empresa en cada tour.
# Añadir varibale gasolina (coste por km recorrido) o variable sueldo (tiempo que tarda en recorrer todo el tour, podría guardarse en una variable el número de veces que va al depósito y añadir un tiempo de espera)
#3)Se podría definir dos tipos de carga, es decir, dos tipos de productos demandados por los clientes.
#4)Definir dos depóstios con productos disponibles ilimitados.

### Planing de cambios para 3 y 4
### 3
### Primero debemos definir los parametros referidos a máxima carga por cada vehículo (max_load 1 y 2), deberemos agregar otro parámetro. Al igual que con max_demand, 1 y 2
### Cambiar las dimensiones del tensor loads, es decir, cambiar dynamic_shape --> num_samples, 2, input_size +1. (No lo tengo muy claro dimensionalidades)
### Crear otra variable demand (demand 1 y demand 2), con la misma dimensionaldad del tensor
### Es probable que se necesite concatenar ambos tensores para crear un único tensor con ambas demandas por cliente y que así tengan las mismas dimensiones el tensor carga por vehículo y el tensor demanda


####4
#### El primer cambio lógico es añadir un depóstio al vector static (script localizaciones fijas)
#### renombrar la variable deposito y añadir deposito2 = torch.tensor([[0],[1]])
#### En el caso del conjunto de entrenamiento expandir el depostio 2 a las dimensiones del tensor cliente y concatenar todos los tensores
#### locations = torch.cat((deposito1.expand(num_samples, -1, -1), deposito2.expand(num_samples, -1, -1), clientes), dim=2)
#### Cambiar las dimensiones del tensor dynamic para que tengas dimensiones (num_samples, 2 , input_size + 2) (El dos corresponde al nº de depósitos)
#### Añadir la hipótesis de que el nuevo depósito comience con la demanda igual a 0 (demands[:, 0, 0] = 0)
#### Aunque haya dos depósitos vamos a obligar al agente a iniciar y finalizar el tour siempre en el depósito 1 (depósito central)
####
#### Actualización de la máscara y tensor dynamic:
####
#### Es probable que haya que cambiar también el hecho de que el agente se deba quedar en el depósito cuando en un minibatch (Instancia del problema) llegue al número de paradas especificado
#### Línea --> 147
