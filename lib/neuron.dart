import 'dart:math';

import 'package:neural_network/layer.dart';
import 'package:dart_numerics/dart_numerics.dart' as numerics;

class Connections {
  double weight;
  double deltaWeight = 0;
  Connections(this.weight);
  void UpdateWeight(double newWeight) {
    weight = newWeight;
  }

  void UpdateDeltaWeight(double newDeltaWeight) {
    deltaWeight = newDeltaWeight;
  }
}

class Bias extends Neuron {
  @override
  double value = 1;
  Bias(super.numberOfOutputs, super.index, super.eta, super.alpha);
}

class Neuron {
  int index;
  int numberOfOutputs;
  double value = 0;
  List<Connections> outputsWeights = [];
  double gradiant = 0;
  double eta; //Neural network training rate
  double alpha; //multiplier of last weight change(momentum)

  Neuron(this.numberOfOutputs, this.index, this.eta, this.alpha) {
    for (var i = 0; i < numberOfOutputs; i++) {
      outputsWeights.add(Connections(Random().nextDouble()));
    }
  }

  void Update(NeuralLayer prevouislayer) {
    for (var i = 0; i < prevouislayer.neurons.length; i++) {
      value += value * prevouislayer.neurons[i].outputsWeights[index].weight;
    }
  }

  static double ActivationFunction(double x) {
    //tanh
    return numerics.tanh(x);
  }

  static double ActivationFunctionDerivative(double x) {
    //tanh derivative
    return 1.0 - (x * x);
  }

  void calculateOutputGradiant(double targetValue) {
    // the goal of this functions is to reduce the error
    double delta = targetValue - value;
    gradiant = delta * Neuron.ActivationFunctionDerivative(value);
  }

  void calculateHiddenGradiant(NeuralLayer nextLayer) {
    double dow = SumDOW(nextLayer);
    gradiant = dow * Neuron.ActivationFunctionDerivative(value);
  }

  void UpdateWeight(NeuralLayer prevLayer) {
    for (var i = 0; i < prevLayer.neurons.length; i++) {
      Neuron prvNeuron = prevLayer.neurons[i];
      double oldDeltaWeight = prvNeuron.outputsWeights[index].deltaWeight;
      double newDeltaWeight =
          //individual input,magified by the gradiant and the train rate
          eta * prvNeuron.value * gradiant
              //also add Momentum = a fraction of the previous delta weight
              +
              alpha * oldDeltaWeight;
      prvNeuron.outputsWeights[index].deltaWeight = newDeltaWeight;
      prvNeuron.outputsWeights[index].weight += newDeltaWeight;
    }
  }

  double SumDOW(NeuralLayer nextLayer) {
    double sum = 0.0;
    for (var i = 0; i < nextLayer.neurons.length; i++) {
      sum += outputsWeights[i].weight * nextLayer.neurons[i].gradiant;
    }
    return sum;
  }
}
