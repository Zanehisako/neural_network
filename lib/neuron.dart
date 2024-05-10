import 'package:neural_network/layer.dart';

class Bias extends Neuron {
  @override
  num value = 1;
  @override
  num weight = 1;
  Bias();
}

class Neuron {
  num value = 0;
  num weight = 0;

  Neuron();

  void Update(NeuralLayer prevouislayer) {
    prevouislayer.neurons.forEach((element) {
      value += element.value * element.weight;
    });
  }
}
