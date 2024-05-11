class Data {
  final List<double> inputs;
  final double targetValue;
  Data({required this.inputs, required this.targetValue});
  void Print() {
    print('$inputs');
    print('targetValue :$targetValue');
  }
}

void ParseData(List<String> contents, List<Data> data) {
  if (data.isEmpty) {
    for (int i = 0; i < contents.length - 1; i += 2) {
      List<String> inputs = [];
      String targetValue;
      List<String> InputTrims = contents[i].trim().split(' ');
      inputs = InputTrims[1].trim().split(',');
      List<String> targetTrims = contents[i + 1].trim().split(' ');
      targetValue = targetTrims.last;
      data.add(Data(
          inputs: inputs.map((e) => double.parse(e)).toList(),
          targetValue: double.parse(targetValue)));
    }
    data.last.Print();
  } else {
    print('data is not empty');
  }
}

class Trainer {
  List<Data> data;

  Trainer({required this.data});
}
