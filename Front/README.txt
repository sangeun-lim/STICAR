`나중에 README.md` 로 파일명 바꿀것

1. 화면 전환
1-1. 설치 모듈
npm install @react-navigation/native 
npm install @react-navigation/stack 
npm install react-native-safe-area-context 
npm install react-native-gesture-handler 
npm install react-native-screens

1-1-1. 경고 메시지 발생 아래 명령어 입력 권장
npx expo install react-native-gesture-handler@~2.5.0 react-native-safe-area-context@4.3.1 react-native-screens@~3.15.0

1-2. 로그인을 위한 모듈
npm install react-native-paper@5.0.0-rc.5

1-3. 하단 메뉴바를 위한 모듈
npm install @react-navigation/bottom-tabs
npm install @expo/vector-icons

1-4. 각 화면에서 해더 Text 지우는 방법
<Stack.Navigator screenOptions={{ headerShown: false }}>
안에 있는 장면들 모두 해더 제거

<Stack.Screen options={{headerShown: false}} name='Begin' component={begin_screen} />
해당 장면의 해더만 제거

1-5. 카메라 사용
npx expo install expo-camera

1-6. 찍은 사진을 저장하기 위해 사용
npx expo install expo-media-library

1-7. 상단 메뉴바를 위해 사용
npm install react-native-segmented-control-tab