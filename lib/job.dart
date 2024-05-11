class Job {
  double pay;
  double? distance;
  double? targetValue;
  double score = 0;
  double? latitude;
  double? longitude;

  Job(
      {required this.pay,
      this.distance,
      this.targetValue,
      required this.latitude,
      required this.longitude});

  void UpdateTargetValue(double newValue) {
    targetValue = newValue;
  }
}
