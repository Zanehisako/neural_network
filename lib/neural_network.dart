import 'package:flutter/material.dart';
import 'package:neural_network/layer.dart';
import 'package:neural_network/neuron.dart';

class NeuralNetwork {
  List<NeuralLayer> layers = [];
  NeuralNetwork(int numOfneurons1, int numOfneurons2, int numOfneurons3) {
    layers.addAll([
      NeuralLayer(numOfneurons1),
      NeuralLayer(numOfneurons2),
      NeuralLayer(numOfneurons3)
    ]);
  }

  NeuralNetwork.fourLayers(int numOfneurons1, int numOfneurons2,
      int numOfneurons3, int numOfneurons4) {
    layers.addAll([
      NeuralLayer(numOfneurons1),
      NeuralLayer(numOfneurons2),
      NeuralLayer(numOfneurons3),
      NeuralLayer(numOfneurons4)
    ]);
  }
  NeuralNetwork.FiveLayers(int numOfneurons1, int numOfneurons2,
      int numOfneurons3, int numOfneurons4, int numOfneurons5) {
    layers.addAll([
      NeuralLayer(numOfneurons1),
      NeuralLayer(numOfneurons2),
      NeuralLayer(numOfneurons3),
      NeuralLayer(numOfneurons4),
      NeuralLayer(numOfneurons5),
    ]);
  }

  void AddBias(int layerIndex) {
    layers[layerIndex].neurons.add(Bias());
  }

  void PrintNetwork() {
    layers.forEach((element) {
      print(element.neurons.length);
      element.neurons.forEach((element) {
        print("the neuron Value is : ${element.value}");
        print("the neuron Weight is : ${element.weight}");
      });
    });
  }

  void UpdateNetwork() {
    for (var i = 1; i < layers.length; i++) {
      layers[i].neurons.forEach((element) {
        element.Update(layers[i - 1]);
      });
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
    network = NeuralNetwork(2, 3, 1);
    network!.UpdateNetwork();
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
