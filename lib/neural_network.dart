import 'dart:math';

import 'package:flutter/material.dart';
import 'package:neural_network/layer.dart';
import 'package:neural_network/neuron.dart';

class NeuralNetwork {
  List<NeuralLayer> layers = [];
  double error = 0.0;
  NeuralNetwork(int numOfneurons1, int numOfneurons2, int numOfneurons3,
      double eta, double alpha) {
    layers.addAll([
      NeuralLayer(numOfneurons1, numOfneurons2, eta, alpha),
      NeuralLayer(numOfneurons2, numOfneurons3, eta, alpha),
      NeuralLayer(numOfneurons3, 0, 0, 0)
    ]);
  }

  NeuralNetwork.fourLayers(int numOfneurons1, int numOfneurons2,
      int numOfneurons3, int numOfneurons4, double eta, double alpha) {
    layers.addAll([
      NeuralLayer(numOfneurons1, numOfneurons2, eta, alpha),
      NeuralLayer(numOfneurons2, numOfneurons3, eta, alpha),
      NeuralLayer(numOfneurons3, numOfneurons4, eta, alpha),
      NeuralLayer(numOfneurons4, 0, 0, 0)
    ]);
  }
  NeuralNetwork.FiveLayers(
      int numOfneurons1,
      int numOfneurons2,
      int numOfneurons3,
      int numOfneurons4,
      int numOfneurons5,
      double eta,
      double alpha) {
    layers.addAll([
      NeuralLayer(numOfneurons1, numOfneurons2, eta, alpha),
      NeuralLayer(numOfneurons2, numOfneurons3, eta, alpha),
      NeuralLayer(numOfneurons3, numOfneurons4, eta, alpha),
      NeuralLayer(numOfneurons4, numOfneurons5, eta, alpha),
      NeuralLayer(numOfneurons5, 0, 0, 0),
    ]);
  }

  void AddBias(
      int layerIndex, int numnextLayerNeurons, double eta, double alpha) {
    NeuralLayer currentlayer = layers[layerIndex];
    currentlayer.neurons.add(
        Bias(numnextLayerNeurons, currentlayer.neurons.length, eta, alpha));
  }

  void PrintNetwork() {
    layers.forEach((element) {
      print(element.neurons.length);
      element.neurons.forEach((element) {
        print("the neuron Value is : ${element.value}");
        element.outputsWeights.forEach(
          (element) {
            print("the neuron Weight is : ${element.weight}");
            print("the neuron Weight is : ${element.deltaWeight}");
          },
        );
      });
    });
  }

  void GetResults() {
    layers.last.neurons.forEach((element) {
      print("the value of the output is ${element.value}");
    });
  }

  void FeedFoward(List<double> inputValues) {
    assert(inputValues.length == layers[0].neurons.length);
    for (var i = 0; i < inputValues.length; i++) {
      layers[0].neurons[i].value = inputValues[i];
    }

    for (var i = 1; i < layers.length; i++) {
      layers[i].neurons.forEach((element) {
        element.Update(layers[i - 1]);
      });
    }
  }

  void BackPropagitation(List<double> targetValues) {
    //calculate the overall net error (RMS)
    NeuralLayer outputLayer = layers.last;
    error = 0.0;

    for (var i = 0; i < outputLayer.neurons.length; i++) {
      double delta = targetValues[i] - outputLayer.neurons[i].value;
      error += delta * delta;
    }
    error /= outputLayer.neurons.length; // get the average error
    error = sqrt(error); //RMS

    //calculate output layer gradiants
    for (var i = 0; i < outputLayer.neurons.length; i++) {
      outputLayer.neurons[i].calculateOutputGradiant(targetValues[i]);
    }
    //calculate hidden layer gradiants
    for (var i = layers.length - 2; i > 0; i--) {
      NeuralLayer hiddenlayer = layers[i];
      NeuralLayer nextLayer = layers[i + 1];
      for (var i = 0; i < hiddenlayer.neurons.length; i++) {
        hiddenlayer.neurons[i].calculateHiddenGradiant(nextLayer);
      }
    }
    //Update the weights from the output layer to the hidden layer
    for (var i = layers.length - 1; i > 0; i--) {
      NeuralLayer layer = layers[i];
      NeuralLayer prevlayer = layers[i - 1];
      for (var i = 0; i < layer.neurons.length; i++) {
        layer.neurons[i].UpdateWeight(prevlayer);
      }
    }
  }
}

class NeuralNetworkUi extends StatefulWidget {
  @override
  State<NeuralNetworkUi> createState() => _NeuralNetworkUiState();
}

class _NeuralNetworkUiState extends State<NeuralNetworkUi> {
  NeuralLayer? layer;
  NeuralNetwork? network;
  @override
  void initState() {
    network = NeuralNetwork(2, 3, 1, 0.1, 0.5);
    network!.FeedFoward([1, 2]);
    network!.PrintNetwork();
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    // TODO: implement build
    return Scaffold(
      body: Center(child: Text('Imagine a neural network here')),
    );
  }
}
