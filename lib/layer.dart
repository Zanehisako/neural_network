import 'package:neural_network/neuron.dart';

class NeuralLayer {
  List<Neuron> neurons = [];
  NeuralLayer(
      int numOfNeurons, int numOfNextLayerNeurons, double eta, double alpha) {
    for (var i = 0; i < numOfNeurons; i++) {
      neurons.add(Neuron(numOfNextLayerNeurons, i, eta, alpha));
    }
  }
}
