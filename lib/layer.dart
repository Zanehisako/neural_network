import 'dart:math';

import 'package:neural_network/neuron.dart';

class NeuralLayer {
  List<Neuron> neurons = [];
  List<Bias> biases = [];
  NeuralLayer(int numOfNeurons, int numOfbiases, int numOfNextLayerNeurons,
      double eta, double alpha) {
    for (var i = 0; i < numOfNeurons; i++) {
      neurons.add(Neuron(numOfNextLayerNeurons, i, eta, alpha));
    }
    for (var i = 0; i < numOfbiases; i++) {
      biases.add(Bias());
    }
  }
}
