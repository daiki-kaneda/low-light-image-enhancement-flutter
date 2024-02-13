import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:midnight_bright_onnx/low_light_image_enhancer.dart';
import 'package:image/image.dart' as img;
void main() {
  runApp(const MainApp());
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: MyHomePage()
    );
  }
}

class MyHomePage extends StatefulWidget{
  const MyHomePage({super.key});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage>{

  LowLightImageEnhancer? _enhancer;
  bool sessionPrepared = false;
  img.Image? inputImage;
  img.Image? resultImage;

  @override
  void initState() {
    super.initState();
    _enhancer = LowLightImageEnhancer();
    _enhancer!.initModel(LolModelType.sciDifficult)
    .then((_){
      setState(() {
        sessionPrepared = true;
      });
    });

  }

  @override
  void dispose() {
    _enhancer?.dispose();
    super.dispose();
  }

  void cleanResult(){
    setState(() {
      inputImage=null;
      resultImage=null;
    });
    PaintingBinding.instance.imageCache.clear();
  }

  Future<void> pickImageAndInfer()async{
    cleanResult();
    final xFile = await ImagePicker()
    .pickImage(source: ImageSource.gallery);
    if(xFile!=null){
      final bytes = await xFile.readAsBytes();
      final image = img.decodeImage(bytes);
      setState(() {
        if(image!=null) inputImage = img.copyResize(image, width:1280,height: 720,maintainAspect: true);
      });
      if(_enhancer!=null && inputImage!=null){
        resultImage = await _enhancer!.inferImage(inputImage!);
        setState(() {});
      }
    }
    
  }
  @override
  Widget build(BuildContext context) {
    return Scaffold(
        body: SingleChildScrollView(
          child: Column(
            children: [
              if(inputImage!=null)
              Image.memory(img.encodePng(inputImage!)),
              if(resultImage!=null)
              Image.memory(img.encodePng(resultImage!)),
            ],
          ),
        ),
        floatingActionButton: FloatingActionButton(
          onPressed: sessionPrepared ? pickImageAndInfer:null
          ),
      );
  }
}
