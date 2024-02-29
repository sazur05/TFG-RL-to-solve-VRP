import torch

#Esta función tiene de entradas el input_size== num_nodo y el STATIC_SIZE == coordenadas
#Con esta funcion estoy creando un tensor de dos dimensiones, donde se encuentra la cantidad de nodos (input_size) y las dimensiones de las coordenadas.
#Así podremos obtener las coordenadas de los clientes para poder visualizarlas cuando se inicie el proceso de entrenamiento o de test


def generador_coordenadas_clientes(input_size, STATIC_SIZE):
    clientes = torch.rand((STATIC_SIZE, input_size))  # clientes
    deposito = torch.tensor([[0], [0]])  # depósito
    localizaciones = torch.cat((deposito, clientes), dim=1)
    print(localizaciones)
    return localizaciones

 # Ahora debemos crear un tensor que acumule todas las instancias deseadas del problema con nustras posiciones fijas
def generador_static_validación(num_samples, localizaciones):
    locations_repetidas = localizaciones.expand(num_samples, -1, -1)
    return locations_repetidas

##En el tensor de entrenamiento las posiciones de los clientes, para cada instancia del problema, deben ser distintas para que el agente aprenda de manera más eficaz y con un número mayor de problemas con distinto entorno
def generador_static_entrenamiento(num_samples, STATIC_SIZE, input_size):
    deposito = torch.tensor([[0], [0]])
    clientes = torch.rand(num_samples, STATIC_SIZE, input_size)
    #Debemos expandir el tensor depósito a las dimensiones del tensor cliente, es decir, añadir dos dimensiones al depósito, para luego poder concatenar ambos tensores
    locations = torch.cat((deposito.expand(num_samples, -1, -1), clientes), dim=2)
    return locations
locations = generador_static_entrenamiento(10,2,10)
print(locations.shape)
