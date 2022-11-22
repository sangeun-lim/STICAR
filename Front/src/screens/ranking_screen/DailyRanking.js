import { SafeAreaView, StatusBar, StyleSheet, Text, View, ScrollView } from "react-native";
import React from "react";

import CustomRankBox from './CustomRankingBox';

function DailyRanking() {
  return (
    <SafeAreaView style={styles.container}>
      <ScrollView
        contentContainerStyle={{
          alignItems: 'center',
          justifyContent: 'center',
        }}
        style={styles.scrollview_container}>
        <CustomRankBox
          rank="1"
          type="first"
          username="홍길동"
          collect_number="500" />
        <CustomRankBox
          rank="2"
          type="second"
          username="파이리"
          collect_number="400" />
        <CustomRankBox
          rank="3"
          type="third"
          username="꼬부기"
          collect_number="300" />
        <CustomRankBox
          rank="4"
          username="김싸피"
          collect_number="200" />
        <CustomRankBox
          rank="5"
          username="피카츄"
          collect_number="100" />
      </ScrollView>
    </SafeAreaView >
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingVertical: StatusBar.currentHeight,
  },
  scrollview_container: {
    marginVertical: 20,
  },
});

export default DailyRanking;
