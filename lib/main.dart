import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:typed_data';
import 'dart:io';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Currency Detection',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: CurrencyDetectionScreen(),
    );
  }
}

class CurrencyDetectionScreen extends StatefulWidget {
  @override
  _CurrencyDetectionScreenState createState() =>
      _CurrencyDetectionScreenState();
}

class _CurrencyDetectionScreenState extends State<CurrencyDetectionScreen> {
  String _result = "Upload an image for detection";
  bool _isUploading = false;

  Future<void> _uploadImage(File image) async {
    setState(() {
      _isUploading = true;
    });

    final String url = 'http://127.0.0.1:5000/predict'; // Flask server URL

    try {
      final bytes = await image.readAsBytes();
      final base64Image = base64Encode(bytes);

      final response = await http.post(
        Uri.parse(url),
        headers: {"Content-Type": "application/json"},
        body: json.encode({"image": base64Image}),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        setState(() {
          _result = "Result: ${data['result']}\nVariance: ${data['variance']}\nSkew: ${data['skew']}\nKurtosis: ${data['kurtosis']}\nEntropy: ${data['entropy']}";
        });
      } else {
        setState(() {
          _result = "Failed to upload image.";
        });
      }
    } catch (e) {
      setState(() {
        _result = "Error: $e";
      });
    } finally {
      setState(() {
        _isUploading = false;
      });
    }
  }

  // Function to pick an image from the gallery
  Future<void> _pickImage() async {
    // Use image_picker package to pick image from gallery or camera
    // Placeholder code to simulate picking an image
    File image = await pickImageFromGallery();  // Replace with your image picker logic
    if (image != null) {
      _uploadImage(image);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Currency Detection"),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _isUploading
                ? CircularProgressIndicator()
                : ElevatedButton(
                    onPressed: _pickImage,
                    child: Text("Pick Image for Detection"),
                  ),
            SizedBox(height: 20),
            Text(_result, textAlign: TextAlign.center),
          ],
        ),
      ),
    );
  }
}

// Placeholder function for image picking, replace with actual logic
Future<File> pickImageFromGallery() async {
  // Implement actual image picking functionality
  return null;  // Return a file object after picking an image
}