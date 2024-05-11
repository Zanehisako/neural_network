import 'dart:math';

double leakyReLU(double x, {double alpha = 0.01}) {
  return x > 0 ? x : alpha * x;
}

double leakyReLUDerivative(double x, {double alpha = 0.01}) {
  return x > 0 ? 1 : alpha;
}

double sigmoid(double x) {
  return 1 / (1 + exp(-x));
}

double sigmoidDerivative(double x) {
  double s = sigmoid(x);
  return s * (1 - s);
}

List<double> softmax(List<double> x) {
  double maxVal = x.reduce(max);
  List<double> expValues = x.map((e) => exp(e - maxVal)).toList();
  double sumExp = expValues.reduce((value, element) => value + element);
  return expValues.map((e) => e / sumExp).toList();
}

List<List<double>> softmaxDerivative(List<double> softmaxOutput) {
  int n = softmaxOutput.length;
  List<List<double>> derivative =
      List.generate(n, (_) => List<double>.filled(n, 0.0));

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        derivative[i][j] = softmaxOutput[i] * (1.0 - softmaxOutput[i]);
      } else {
        derivative[i][j] = -softmaxOutput[i] * softmaxOutput[j];
      }
    }
  }

  return derivative;
}
