import torch

#Esta función tiene de entradas el input_size== num_nodo y el STATIC_SIZE == coordenadas
#Con esta funcion estoy creando un tensor de dos dimensiones, donde se encuentra la cantidad de nodos (input_size) y las dimensiones de las coordenadas.


def generador_coordenadas_clientes(input_size, STATIC_SIZE):
    clientes = torch.rand((STATIC_SIZE, input_size))  # clientes
    deposito = torch.tensor([[0], [0]])  # depósito
    #deposito2 = torch.tensor([[0], [1]])
    localizaciones = torch.cat((deposito, clientes), dim=1) ##localizaciones = torch.cat( deposito1, deposito2, clientes), dim=2)
    print(localizaciones)
    return localizaciones

 # Ahora debemos crear un tensor que acumule todas las instancias deseadas del problema con nustras posiciones fijas

def generador_static_entrenamiento(num_samples, STATIC_SIZE, input_size):
    deposito = torch.tensor([[0], [0]])
    #deposito2 = torch.tensor([[0], [1]])
    clientes = torch.rand(num_samples, STATIC_SIZE, input_size)
    #Debemos expandir el tensor depósito a las dimensiones del tensor cliente, es decir, añadir dos dimensiones al depósito, para luego poder concatenar ambos tensores
    locations = torch.cat((deposito.expand(num_samples, -1, -1), clientes), dim=2) #locations = torch.cat((deposito.expand(num_samples, -1, -1), deposito2.expand(num_samples, -1, -1), clientes), dim=2)
    return locations


def generador_static_validación(num_samples, localizaciones):
    locations_repetidas = localizaciones.expand(num_samples, -1, -1)
    return locations_repetidas
###Con estas dos funciones podremos crear unas localizaciones fijas de unos clientes que se repitan tanto en las instancias de entrenamiento, como de validación, como de test.

### 3er cambio: 2 depósitos en el problema, al igual que en el caso anterior debemos determinar también cuál será la posición determinada de nuestro depósito.
### Por otro lado, también deberemos cambiar la dimensionalidad de locations1 = torch.rand((input_size-1, STATIC_SIZE)) --> locations1 = torch.rand((input_size-2, STATIC_SIZE))

locations = generador_static_entrenamiento(10,2,10)
print(locations.shape)