import 'dart:math';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:neural_network/FindBestJob.dart';
import 'package:neural_network/job.dart';
import 'package:neural_network/layer.dart';
import 'package:neural_network/neuron.dart';
import 'package:neural_network/trainingData.dart';

class NeuralNetwork {
  List<NeuralLayer> layers = [];
  double error = 0.0;
  double averageError = 0.0;
  NeuralNetwork(int numOfneurons1, int numOfneurons2, int numOfneurons3,
      double eta, double alpha) {
    layers.addAll([
      NeuralLayer(numOfneurons1, numOfneurons2, eta, alpha),
      NeuralLayer(numOfneurons2, numOfneurons3, eta, alpha),
      NeuralLayer(numOfneurons3, 0, eta, alpha)
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

  void PrintNetwork(int datalength) {
    layers.forEach((element) {
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
  }

  void GetResults() {
    layers.last.neurons.forEach((element) {
      print("the network output is ${element.myValue}");
    });

    print('the error is $error');
  }

  void FeedFoward(List<double> inputValues) {
    NeuralLayer prevLayer;

    for (var i = 0; i < inputValues.length - 1; i++) {
      layers[0].neurons[i].myValue = inputValues[i];
    }
    for (var i = 1; i < layers.length; i++) {
      prevLayer = layers[i - 1];
      for (var j = 0; j < layers[i].neurons.length; j++) {
        layers[i].neurons[j].FeedFoward(prevLayer, i);
      }
    }
  }

  void BackPropagitation(List<double> targetValues) {
    //calculate the overall net error (RMS)
    NeuralLayer outputLayer = layers.last;
    error = 0.0;

    for (var i = 0; i < outputLayer.neurons.length; i++) {
      double delta = targetValues[i] - outputLayer.neurons[i].myValue;
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
    averageError += error;
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
  NeuralNetwork network = NeuralNetwork(2, 3, 1, 0.3, 0.5);
  File dataFile = File('Data.txt');
  List<Data> data = [];
  Trainer? trainer;
  bool gettingResult = true;
  double? result;
  List<String> contents = [];

  @override
  void initState() {
    /*for (double i = 0; i < 10000000; i++) {
      jobs.add(Job(pay: i, latitude: i, longitude: i));
    }

    jobs = FindBestJob1(jobs);
    for (var i = 0; i < jobs.length; i++) {
      network.FeedFoward([jobs[i].pay, jobs[i].distance!]);
      network.BackPropagitation([jobs[i].score!]);
      /*print('');
      print('the inputs are : ${jobs[i].pay} ${jobs[i].distance}');
      print('the target value is : ${jobs[i].score}');
      network.GetResults();*/
    }
    print('finished');
    network.PrintNetwork(jobs.length);*/
    setState(() {
      contents = dataFile.readAsLinesSync();
    });

    print(contents);

    //ParseData(contents, data);
    print(data.length);
    /*data.forEach((element) {
      element.Print();
    });*/

    super.initState();
  }

  void NetworkResult(double pay, double distance) {
    print('the inputs are ${pay} ${distance}');
    network.PrintNetwork(jobs.length);
    network.FeedFoward([pay, distance]);
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
        padding: const EdgeInsets.all(250),
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
            TextButton(
                onPressed: () {
                  /*NetworkResult(double.parse(payController.text),
                      double.parse(distanceController.text));
                  network.PrintNetwork(jobs.length);*/
                  ParseData(contents, data);
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
