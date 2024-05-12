import 'dart:math';

import 'package:neural_network/activationFunctions.dart';
import 'package:neural_network/layer.dart';
import 'package:dart_numerics/dart_numerics.dart' as numerics;

enum ActivationFunctions {
  sigmoid,
  tanh,
  leakyReLU,
}

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

class Bias {
  double value = Random().nextDouble();
  double gradiant = 0;
  Bias();
  void calculateOutputGradiant(double targetValue) {
    // the goal of this functions is to reduce the error
    double delta = targetValue - value;
    gradiant = delta *
        Neuron.ActivationFunctionDerivative(value, ActivationFunctions.tanh);
  }

  void calculateHiddenGradiant(double targetValue, NeuralLayer layer) {
    // the goal of this functions is to reduce the error
    double delta = targetValue - value;
    gradiant = delta *
        Neuron.ActivationFunctionDerivative(value, ActivationFunctions.tanh);
  }
}

class Neuron {
  double error = 0;
  int index;
  int numberOfOutputs;
  double _myValue = Random().nextDouble();
  List<Connections> outputsWeights = [];
  double gradiant = 0;
  double eta; //Neural network learning rate
  double alpha; //multiplier of last weight change(momentum)

  Neuron(this.numberOfOutputs, this.index, this.eta, this.alpha) {
    for (var i = 0; i < numberOfOutputs; i++) {
      outputsWeights.add(Connections(Random().nextDouble()));
    }
  }
  double get myValue => _myValue;

  set myValue(double value) {
    _myValue = value;
  }

  void FeedFoward(NeuralLayer prevLayer, int layerIndex, double bias) {
    double sum = 0.0;
    for (var i = 0; i < prevLayer.neurons.length; i++) {
      sum += ((prevLayer.neurons[i].myValue *
              prevLayer.neurons[i].outputsWeights[index].weight) +
          bias);
    }

    switch (layerIndex) {
      case 1:
        myValue = ActivationFunction(sum, ActivationFunctions.tanh);
        break;
      case 2:
        myValue = ActivationFunction(sum, ActivationFunctions.tanh);
        break;
      default:
    }
  }

  static double ActivationFunction(
      double x, ActivationFunctions activationFunction) {
    switch (activationFunction) {
      case ActivationFunctions.sigmoid:
        return sigmoid(x);
      case ActivationFunctions.tanh:
        return numerics.tanh(x);
      case ActivationFunctions.leakyReLU:
        return leakyReLU(x);
    }
  }

  static double ActivationFunctionDerivative(
      double x, ActivationFunctions activationFunction) {
    switch (activationFunction) {
      case ActivationFunctions.sigmoid:
        return sigmoidDerivative(x);
      case ActivationFunctions.tanh:
        return 1 - (numerics.tanh(x) * numerics.tanh(x));
      case ActivationFunctions.leakyReLU:
        return leakyReLUDerivative(x);
    }
  }

  void calculateOutputGradiant(double targetValue) {
    // the goal of this functions is to reduce the error
    double delta = targetValue - myValue;
    gradiant = delta *
        Neuron.ActivationFunctionDerivative(myValue, ActivationFunctions.tanh);
  }

  void calculateHiddenGradiant(NeuralLayer nextLayer) {
    double dow = SumDOW(nextLayer);
    gradiant = dow *
        Neuron.ActivationFunctionDerivative(myValue, ActivationFunctions.tanh);
  }

  void UpdateWeight(NeuralLayer prevLayer) {
    for (var i = 0; i < prevLayer.neurons.length; i++) {
      Neuron prvNeuron = prevLayer.neurons[i];
      double oldDeltaWeight = prvNeuron.outputsWeights[index].deltaWeight;
      double newDeltaWeight =
          //individual input,magified by the gradiant and the train rate
          eta * prvNeuron.myValue * gradiant
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
