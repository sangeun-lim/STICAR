import React, { useState, useRef, useEffect } from 'react'
import { StyleSheet, Text, View, TouchableOpacity, SafeAreaView, Image } from 'react-native';
import { Camera, CameraType } from 'expo-camera'; // 카메라 라이브러리
import * as MediaLibrary from 'expo-media-library'; // 저장소 라이브러리
import { useIsFocused } from '@react-navigation/native';

import S3 from 'aws-sdk/clients/s3';
import * as FileSystem from 'expo-file-system';
import { decode } from "base64-arraybuffer";

// 카메라 촬영 및 플래시 모드 버튼 이미지
import Camera_Icon from '../../../assets/camera_images/camera_not_fill.png';
import Flash_Off from '../../../assets/camera_images/flash_off.png';
import Flash_On from '../../../assets/camera_images/flash_on.png';
import Camera_Reverse from '../../../assets/camera_images/camera_not_reverse.png';
import Retake_Pic from '../../../assets/camera_images/retake_pic.png';
import Save_Pic from '../../../assets/camera_images/save_pic.png';
import Upload_Pic from '../../../assets/camera_images/upload_pic.png';

// 뒤로가기 버튼
import Back_Icon from '../../../assets/images/back_white.png';

function AppCamera({ navigation }) {
  const [hasCameraPermission, setHasCameraPermission] = useState(); // 카메라 접근 권한
  const [photo, setPhoto] = useState(); // 이미지 파일
  const [type, setType] = useState(Camera.Constants.Type.back); // 전면, 후면 카메라
  const [flash, setFlash] = useState(Camera.Constants.FlashMode.off); // 플래시 모드
  const cameraRef = useRef();

  const [hasMediaLibraryPermission, setHasMediaLibraryPermission] = useState(); // 저장소 접근 권한

  const isFocused = useIsFocused();

  useEffect(() => {
    const StartGetPermssion = async () => {
      const CameraPermission = await Camera.requestCameraPermissionsAsync();
      const MediaLibraryPermission = await MediaLibrary.requestPermissionsAsync();

      setHasCameraPermission(CameraPermission.status === "granted");
      setHasMediaLibraryPermission(MediaLibraryPermission.status === "granted");
    }

    StartGetPermssion();
  }, []);

  const takePic = async () => {
    let options = {
      base64: true,
      imageType: '.jpg',
      exif: false,
      quality: 1,
    };

    if (cameraRef) {
      try {
        const newPhoto = await cameraRef.current.takePictureAsync(options);
        setPhoto(newPhoto);
      } catch (e) {
        console.log(e);
      }
    }
  };

  if (photo) { // 촬영한 사진이 있는 경우
    MediaLibrary.saveToLibraryAsync(photo.uri);

    let retake_photo = () => {
      setPhoto(undefined);
    };

    let send_photo = async () => {
      MediaLibrary.createAssetAsync(photo.uri).then(asset => {
        MediaLibrary.getAssetInfoAsync(asset.id).then(info => {
          console.log(info);
          const file = {
            uri: '',
            type: 'image/jpeg',
            name: 'test',
          }

          file.uri = info.uri;
          file.name = info.filename;
          file.type = 'image/jpeg';

          uploadImageOnS3(file);
        })
      }).catch(error => {
        this.props.setError(error.toString());
      });
    }

    const uploadImageOnS3 = async (file) => {
      const s3bucket = new S3({
        accessKeyId: 'AKIA347XUSDPZ3XSTFAM',
        secretAccessKey: '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',
        Bucket: 'sticar-camera-picture',
        signatureVersion: 'v4',
      });

      let contentType = 'image/jpeg';
      let contentDeposition = 'inline;filename="' + file.name + '"';
      const base64 = await FileSystem.readAsStringAsync(photo.uri, { encoding: 'base64' });
      // console.log(base64);
      const arrayBuffer = decode(base64);

      s3bucket.createBucket(() => {
        const params = {
          Bucket: 'sticar-camera-picture',
          Key: file.name,
          Body: arrayBuffer,
          ContentDisposition: contentDeposition,
          ContentType: contentType,
        };

        s3bucket.upload(params, (err, data) => {
          if (err) {
            console.log('error in callback');
          }
          else {
            console.log('success');
            console.log("Respomse URL : " + data.Location);
            setPhoto(undefined);
          }
        });
      });
    };

    return (
      <SafeAreaView style={styles.showImage}>
        <Image style={styles.preview} source={{ uri: "data:image/jpg;base64," + photo.base64 }} />
        <View style={styles.Save_or_Not_buttons}>
          <TouchableOpacity
            style={styles.take_pic_button}
            onPress={retake_photo} >
            <Image
              style={styles.button_image}
              source={Retake_Pic} />
          </TouchableOpacity>
          {hasMediaLibraryPermission ?
            <TouchableOpacity
              style={styles.take_pic_button}
              onPress={send_photo} >
              <Image
                style={styles.button_image}
                source={Upload_Pic} />
            </TouchableOpacity>
            : undefined
          }
        </View>
      </SafeAreaView>
    );
  };

  if (hasCameraPermission === undefined) {
    return (
      <View style={styles.warning_container}>
        <Text>No access to Camera</Text>
      </View>
    );
  }
  else if (hasCameraPermission === false) {
    return (
      <View style={styles.get_permission_container}>
        <Text style={styles.get_permission_text}>Permission for camera not granted.</Text>
        <TouchableOpacity
          style={styles.get_permission_button}
          onPress={setHasCameraPermission}>
          <Text style={styles.get_permission_text}>Get Premisson</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (isFocused == true) {
    return (
      <Camera
        style={styles.camera}
        type={type}
        flashMode={flash}
        ref={cameraRef} >
        <TouchableOpacity
          style={styles.back_icon}
          onPress={() => navigation.navigate("Home")}>
          <Image
            style={styles.back_icon_image}
            source={Back_Icon} />
        </TouchableOpacity>
        <View
          style={styles.car_pic_area}
        />
        <View style={styles.take_pic_button_container}>
          <TouchableOpacity
            style={styles.take_pic_button}
            onPress={() => {
              setFlash(flash === Camera.Constants.FlashMode.off ? Camera.Constants.FlashMode.on : Camera.Constants.FlashMode.off)
            }}>
            <Image
              style={styles.button_image}
              source={(flash === Camera.Constants.FlashMode.off ? Flash_Off : Flash_On)} />
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.take_pic_button}
            onPress={takePic}>
            <Image
              style={styles.button_image}
              source={Camera_Icon} />
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.take_pic_button}
            onPress={() => {
              setType(type === CameraType.back ? CameraType.front : CameraType.back)
            }}>
            <Image
              style={styles.button_image}
              source={Camera_Reverse} />
          </TouchableOpacity>
        </View>
      </Camera>
    );
  }
  else {
    return null;
  }
};

const styles = StyleSheet.create({
  camera: {
    flex: 1,
    alignItems: 'center',
  },
  car_pic_area: {
    position: 'absolute',
    top: "15%",
    width: "70%",
    height: "65%",
    backgroundColor: 'transparent',
    borderColor: 'white',
    borderWidth: 3,
  },
  take_pic_button_container: {
    flexDirection: 'row',
    width: '100%',
    position: 'absolute',
    bottom: 0,
    justifyContent: 'space-evenly',
  },
  take_pic_button: {
    width: '15%',
    maxWidth: 60,
    aspectRatio: 1,
    maxHeight: 60,
    backgroundColor: '#fff',
    borderRadius: 30,
    opacity: 0.66,
    alignItems: 'center',
    justifyContent: 'center',
    marginVertical: 30,
    padding: 3,
  },
  back_icon: {
    position: 'absolute',
    top: "5%",
    left: "5%",
    width: '10%',
    maxWidth: 60,
    aspectRatio: 1,
    maxHeight: 60,
    borderRadius: 30,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 3,
  },
  back_icon_image: {
    width: '100%',
    height: '100%',
    resizeMode: 'contain'
  },
  button_image: {
    width: '100%',
    height: '100%',
    resizeMode: 'contain'
  },

  warning_container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  get_permission_container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  get_permission_text: {
    fontSize: 20,
    fontStyle: 'italic',
  },
  get_permission_button: {
    width: '50%',
    height: '8%',
    maxWidth: 210,
    maxHeight: 60,
    backgroundColor: '#0C9AF2',
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    marginVertical: 40,
    padding: 5,
  },

  showImage: {
    flex: 1,
  },
  Save_or_Not_buttons: {
    position: 'absolute',
    width: "100%",
    bottom: 0,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-evenly',
  },
  preview: {
    alignSelf: 'stretch',
    flex: 1
  },
});

export default AppCamera;
