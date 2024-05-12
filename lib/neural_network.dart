import 'dart:math';
import 'dart:io';

import 'package:flutter/material.dart';

import 'package:neural_network/job.dart';
import 'package:neural_network/layer.dart';
import 'package:neural_network/neuron.dart';
import 'package:neural_network/trainingData.dart';

class NeuralNetwork {
  List<NeuralLayer> layers = [];
  double error = 0.0;
  double averageError = 0.0;
  NeuralNetwork(
      int numOfneurons1,
      int numOfBiases1,
      int numOfneurons2,
      int numOfBiases2,
      int numOfneurons3,
      int numOfBiases3,
      double eta,
      double alpha) {
    layers.addAll([
      NeuralLayer(numOfneurons1, numOfBiases1, numOfneurons2, eta, alpha),
      NeuralLayer(numOfneurons2, numOfBiases2, numOfneurons3, eta, alpha),
      NeuralLayer(numOfneurons3, numOfBiases3, 0, eta, alpha)
    ]);
  }

  NeuralNetwork.fourLayers(
      int numOfneurons1,
      int numOfBiases1,
      int numOfneurons2,
      int numOfBiases2,
      int numOfneurons3,
      int numOfBiases3,
      int numOfneurons4,
      int numOfBiases4,
      double eta,
      double alpha) {
    layers.addAll([
      NeuralLayer(numOfneurons1, numOfBiases1, numOfneurons2, eta, alpha),
      NeuralLayer(numOfneurons2, numOfBiases2, numOfneurons3, eta, alpha),
      NeuralLayer(numOfneurons3, numOfBiases3, numOfneurons4, eta, alpha),
      NeuralLayer(numOfneurons4, numOfBiases4, 0, 0, 0)
    ]);
  }
  NeuralNetwork.FiveLayers(
      int numOfneurons1,
      int numOfBiases1,
      int numOfneurons2,
      int numOfBiases2,
      int numOfneurons3,
      int numOfBiases3,
      int numOfneurons4,
      int numOfBiases4,
      int numOfneurons5,
      int numOfBiases5,
      double eta,
      double alpha) {
    layers.addAll([
      NeuralLayer(numOfneurons1, numOfBiases1, numOfneurons2, eta, alpha),
      NeuralLayer(numOfneurons2, numOfBiases2, numOfneurons3, eta, alpha),
      NeuralLayer(numOfneurons3, numOfBiases3, numOfneurons4, eta, alpha),
      NeuralLayer(numOfneurons4, numOfBiases4, numOfneurons5, eta, alpha),
      NeuralLayer(numOfneurons5, numOfBiases5, 0, 0, 0),
    ]);
  }

  void PrintNetwork(int datalength) {
    layers.forEach((element) {
      print('---------------------------');
      print(element.neurons.length);
      element.neurons.forEach((element) {
        print("the neuron Value is : ${element.myValue}");
        element.outputsWeights.forEach(
          (element) {
            print("the neuron Weight is : ${element.weight}");
            print("the neuron delta Weight is : ${element.deltaWeight}");
          },
        );
        print("the gradiant is : ${element.gradiant}");
      });
    });
    print(
        'The average Error of the network was : ${averageError / datalength}');
    print('---------------------------');
  }

  void GetResults() {
    layers.last.neurons.forEach((element) {
      print("the network output is ${element.myValue}");
    });

    print('the error is $error');
  }

  void FeedFoward(List<double> inputValues) {
    NeuralLayer prevLayer;

    for (var i = 0; i < inputValues.length; i++) {
      layers[0].neurons[i].myValue = inputValues[i];
    }
    for (var i = 1; i < layers.length; i++) {
      prevLayer = layers[i - 1];
      for (var j = 0; j < layers[i].neurons.length; j++) {
        layers[i]
            .neurons[j]
            .FeedFoward(prevLayer, i, layers[i].biases[j].value);
      }
    }
  }

  void BackPropagitation(List<double> targetValues) {
    //calculate the overall net error (RMS)
    NeuralLayer outputLayer = layers.last;
    double networkError = 0.0;
    for (var i = 0; i < outputLayer.neurons.length; i++) {
      double delta = targetValues[i] - outputLayer.neurons[i].myValue;
      double error = delta * delta; // calculate squared error for this neuron
      outputLayer.neurons[i].error = error;
      networkError += error; // accumulate squared error for the network
    }
    networkError /= outputLayer.neurons.length; // get the average networkError
    networkError = sqrt(networkError); // RMS

    //calculate output layer gradients
    for (var i = 0; i < outputLayer.neurons.length; i++) {
      outputLayer.neurons[i].calculateOutputGradiant(targetValues[i]);
    }
    //calculate hidden layer gradients
    for (var i = layers.length - 2; i > 0; i--) {
      NeuralLayer hiddenLayer = layers[i];
      NeuralLayer nextLayer = layers[i + 1];
      for (var j = 0; j < hiddenLayer.neurons.length; j++) {
        hiddenLayer.neurons[j].calculateHiddenGradiant(nextLayer);
      }
    }

    // Update the biases for the hidden layers
    for (var i = 1; i < layers.length; i++) {
      NeuralLayer layer = layers[i];
      for (var j = 0; j < layer.biases.length; j++) {
        layer.biases[j].gradiant = Neuron.ActivationFunctionDerivative(
            layer.biases[j].value, ActivationFunctions.sigmoid);
      }
    }

    // Update the weights from the output layer to the hidden layer
    for (var i = layers.length - 1; i > 0; i--) {
      NeuralLayer layer = layers[i];
      NeuralLayer prevLayer = layers[i - 1];
      for (var j = 0; j < layer.neurons.length; j++) {
        layer.neurons[j].UpdateWeight(prevLayer);
      }
    }
    averageError += networkError;
  }
}

class NeuralNetworkUi extends StatefulWidget {
  @override
  State<NeuralNetworkUi> createState() => _NeuralNetworkUiState();
}

class _NeuralNetworkUiState extends State<NeuralNetworkUi> {
  List<Job> jobs = [];
  TextEditingController payController = TextEditingController();
  TextEditingController distanceController = TextEditingController();
  TextEditingController xController = TextEditingController();
  TextEditingController yController = TextEditingController();
  NeuralNetwork network = NeuralNetwork(2, 0, 3, 3, 1, 1, 0.1, 0.1);
  File dataFile = File('Data.txt');
  File xorDataFile = File('Xor_Dataset.txt');
  List<Data> data = [];
  Trainer? trainer;
  bool gettingResult = true;
  double? result;
  List<String> contents = [];

  @override
  void initState() {
    setState(() {
      contents = xorDataFile.readAsLinesSync();
    });
    ParseDataXOR(contents, data);
    //ParseData(contents, data);
    //network.PrintNetwork(data.length);
    for (var i = 0; i < data.length; i++) {
      network.FeedFoward(data[i].inputs);
      network.BackPropagitation([data[i].targetValue]);
      print('');
      print('the inputs are : ${data[i].inputs.first} ${data[i].inputs.last}');
      print('the target value is : ${data[i].targetValue}');
      network.GetResults();
    }
    /*print('Before feedFoward');
    network.PrintNetwork(data.length);
    print('the inputs are :');
    print(data.first.inputs);
    network.FeedFoward(data.first.inputs);
    print('After feedFoward/before BackPropagation');
    network.PrintNetwork(data.length);
    network.BackPropagitation([data.first.targetValue]);
    print('After Back Propagition');
    //network.GetResults();
    network.PrintNetwork(data.length);
    network.FeedFoward(data[1].inputs);
    network.PrintNetwork(data.length);*/
    super.initState();
  }

  void NetworkResult(double pay, double distance) {
    print('the inputs are ${pay} ${distance}');
    network.FeedFoward([pay, distance]);
    setState(() {
      result = network.layers.last.neurons[0].myValue;
      gettingResult = false;
    });
  }

  void XORnetwork(double x, double y) {
    print('the inputs are ${x} ${y}');
    network.FeedFoward([x, y]);
    setState(() {
      result = network.layers.last.neurons[0].myValue;
      gettingResult = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
          child: Container(
        padding: const EdgeInsets.all(50),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            const Text('Enter a job :'),
            TextField(
              controller: payController,
              decoration: const InputDecoration(hintText: 'Pay'),
            ),
            TextField(
              controller: distanceController,
              decoration: const InputDecoration(hintText: 'distance'),
            ),
            const SizedBox(
              height: 15,
            ),
            const Text('XOR network :'),
            TextField(
              controller: xController,
              decoration: const InputDecoration(hintText: 'X'),
            ),
            TextField(
              controller: yController,
              decoration: const InputDecoration(hintText: 'Y'),
            ),
            TextButton(
                onPressed: () {
                  /*NetworkResult(double.parse(payController.text),
                      double.parse(distanceController.text));
                  */
                  double x = double.parse(xController.text);
                  double y = double.parse(yController.text);
                  XORnetwork(x, y);
                  network.PrintNetwork(jobs.length);

                  print(data.length);
                },
                child: const Text('Start')),
            gettingResult ? SizedBox() : Text('$result'),
          ],
        ),
      )),
    );
  }
}
