// ignore_for_file: non_constant_identifier_names

import 'package:latlong2/latlong.dart';

import 'job.dart';

import 'package:geolocator/geolocator.dart';

void ComparePay1(List<Job> jobs) {
  jobs.sort((a, b) => a.pay.compareTo(b.pay));

  num max = jobs[jobs.length - 1].pay;
  jobs[jobs.length - 1].score += 0.6;

  for (int i = 0; i < jobs.length; i++) {
    if (jobs[i].pay == max) {
      jobs[i].score += ((i / jobs.length) * 0.6);
    } else {
      jobs[i].score += ((i / jobs.length) * 0.6);
    }
  }
}

void CompareLocation1(List<Job> jobs) {
  LatLng current = LatLng(34.86951789413617, -1.322405198440011);

  jobs.forEach((element) {
    if (element.latitude == null) {
      element.distance = 0;
      return;
    }
    final distance = Geolocator.distanceBetween(current.latitude,
        current.longitude, element.latitude!, element.longitude!);

    element.distance = distance;
  });
  jobs.sort((a, b) => a.distance!.compareTo(b.distance!));

  num max = jobs[0].distance!;
  jobs[0].score += 0.4;

  for (int i = 0; i < jobs.length; i++) {
    if (max == jobs[i].distance) {
      jobs[i].score += (((jobs.length - 0) / jobs.length) * 0.4);
    } else {
      jobs[i].score += ((((jobs.length - (i + 1)) / jobs.length)) * 0.4);
    }
  }
}

List<Job> FindBestJob1(List<Job> jobs) {
  ComparePay1(jobs);
  CompareLocation1(jobs);

  jobs.forEach((element) {
    if (element.score >= 0.5) {
      element.targetValue = 1;
    } else {
      element.targetValue = 0;
    }
  });

  return jobs;
}
