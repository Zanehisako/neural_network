import 'package:neural_network/neuron.dart';

class NeuralLayer {
  List<Neuron> neurons = [];
  NeuralLayer(int numOfNeurons) {
    neurons = List<Neuron>.filled(numOfNeurons, Neuron());
  }
  
}
