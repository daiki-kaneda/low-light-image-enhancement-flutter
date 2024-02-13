
import 'dart:developer';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:image/image.dart' as img;

enum LolModelType {
  agllNet,
  bread,
  pairlie,
  sciEasy,
  sciMedium,
  sciDifficult,
  zeroDCE,
}

class LowLightImageEnhancer {
  OrtSession? _session;
  OrtSessionOptions? _options;
  LolModelType _type = LolModelType.zeroDCE;
  String get _modelPath => _getModelPath(_type);
  // TODO : adapt _inputWidth and _inputHeight to _type like _modelPath
  int get _inputWidth => 1280;
  int get _inputHeight => _type==LolModelType.agllNet ? 768:720;

  LowLightImageEnhancer() {
    // onnx runtimeの環境を初期化
    OrtEnv.instance.init();
  }

  void dispose() {
    // 各セッション、オプション、環境を解放
    _session?.release();
    _options?.release();
    OrtEnv.instance.release();
  }

  void release() {
    // 各セッション、オプション
    _session?.release();
    _options?.release();
  }

  Future<void> initModel(LolModelType type) async {
    // release current session
    release();
    // get new modelpath
    _type = type;
    log("$_type");
    // set OrtSessionOptions
    _options = OrtSessionOptions()
      ..setInterOpNumThreads(1)
      ..setIntraOpNumThreads(1)
      ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll);
    // set session
    final rawData = await rootBundle.load(_modelPath);
    final bytes = rawData.buffer.asUint8List();
    _session = OrtSession.fromBuffer(bytes, _options!);
  }

  String _getModelPath(LolModelType type) {
    // TODO: implement "return modelPath"
    switch (type) {
      case LolModelType.agllNet:
        return "assets/models/agllnet_768*1280.onnx";
      case LolModelType.bread:
        return "assets/models/bread_720x1280.onnx";
      case LolModelType.pairlie:
        return "assets/models/pairlie_720x1280.onnx";
      case LolModelType.sciEasy:
        return "assets/models/sci_easy_720x1280.onnx";
      case LolModelType.sciMedium:
        return "assets/models/sci_medium_720x1280.onnx";
      case LolModelType.sciDifficult:
        return "assets/models/sci_difficult_720x1280.onnx";
      case LolModelType.zeroDCE:
        return "assets/models/zero_dce_720*1280.onnx";
    }
  }

  Future<img.Image> inferImage(img.Image image)async{
    if(_session!=null){
      final inputName = _session!.inputNames[0];
      // preprocess image
      // TODO: define preprocess function which is adapted to _type
      final inputImage = img.copyResize(
        image,
        width: _inputWidth,
        height: _inputHeight,
        maintainAspect: true,
        interpolation: img.Interpolation.average
      );
      // shape:(1,3,720,1280)
      final inputTensor = [
        List.generate(3,
         (i) => List.generate(
          _inputHeight, (y) => Float32List.fromList(List.generate(
            _inputWidth, (x) {
              final pixel = inputImage.getPixel(x, y);
              return [
                pixel.r/255,
                pixel.g/255,
                pixel.b/255,
              ][i];
            }
            ))
          )
         )
      ];

      final inputOrt = OrtValueTensor.createTensorWithDataList(
        inputTensor,[1,3,_inputHeight,_inputWidth]
      );
      final runOptions = OrtRunOptions();
      final inputs = {inputName:inputOrt};
      final List<OrtValue?>? outputs = await _session!.runAsync(runOptions, inputs);
      if(outputs!=null){
        // TODO: adapt outputShape to _type
        List<List<List<double>>>? outputTensor; // shape:(720,1280,3)
        if([
          LolModelType.bread,
          LolModelType.pairlie,
          LolModelType.sciEasy,
          LolModelType.sciMedium,
          LolModelType.sciDifficult,
          ]
          .contains(_type)
          ){
          // output shape: (3,720,1280)
         final output = (outputs[0]?.value as List<List<List<List<double>>>>)[0];
         outputTensor = List.generate(
          _inputHeight, (y) => List.generate(
            _inputWidth,(x) => [
              output[0][y][x],
              output[1][y][x],
              output[2][y][x],
            ]
          )
         );
        }else if(_type==LolModelType.agllNet){
          // output shape: (768,1280,10)
         final output = (outputs[0]?.value as List<List<List<List<double>>>>)[0];
         outputTensor = List.generate(
          _inputHeight, (y) => List.generate(
            _inputWidth,(x) => [
              output[y][x][4].clamp(0,1),
              output[y][x][5].clamp(0,1),
              output[y][x][6].clamp(0,1),
            ]
          )
         );
        }else{
          outputTensor = outputs[0]?.value as List<List<List<double>>>;
        }
        img.Image outputImage = img.Image(
          width: _inputWidth,
          height: _inputHeight
        );
        for(int y=0;y<_inputHeight;y++){
          for(int x=0;x<_inputWidth;x++){
            if(_type==LolModelType.agllNet){
            outputImage.setPixel(
              x,
              y,
              img.ColorUint8.rgb(
                (outputTensor[y][x][0]*255).toInt().clamp(0, 255),
                (outputTensor[y][x][1]*255).toInt().clamp(0, 255),
                (outputTensor[y][x][2]*255).toInt().clamp(0, 255),
              )
               );
            }else{
              outputImage.setPixel(
              x,
              y,
              img.ColorUint8.rgb(
                (outputTensor[y][x][0]).toInt().clamp(0, 255),
                (outputTensor[y][x][1]).toInt().clamp(0, 255),
                (outputTensor[y][x][2]).toInt().clamp(0, 255),
              )
               );
            }
          }
        }
        inputOrt.release();
        runOptions.release();
        for(final value in outputs){
          value?.release();
        }
        return outputImage;
      }else{
        log("outputs is empty");
        return image;
      }
    }else{
      log("session is empty");
      return image;
    }
  }
}
