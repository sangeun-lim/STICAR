import { View, Text, StyleSheet, Image } from 'react-native'
import React from 'react'

import First_Icon from '../../../assets/rank_images/First.png'
import Second_Icon from '../../../assets/rank_images/Second.png'
import Third_Icon from '../../../assets/rank_images/Third.png'

const RankingBox = ({ rank, user_pic, username, collect_number, type = "under" }) => {
  return (
    <View style={[styles.container, styles[`container_${type}`]]}>
      {rank === '1'
        ? <Image
          style={{
            width: '10%',
            height: '100%',
            resizeMode: 'contain',
          }}
          source={First_Icon} />
        : (rank === '2'
          ? <Image
            style={{
              width: '10%',
              height: '100%',
              resizeMode: 'contain',
            }}
            source={Second_Icon} />
          : (rank === '3'
            ? <Image
              style={{
                width: '10%',
                height: '100%',
                resizeMode: 'contain'
              }}
              source={Third_Icon} />
            : <Text style={styles.text}> {rank}</Text>
          )
        )
      }
      <Text>{username}</Text>
      <Text>{collect_number}</Text>
    </View >
  )
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    width: '90%',
    borderColor: 'black',
    borderWidth: 1,
    borderRadius: 10,
    padding: 10,
    margin: 5,
    justifyContent: "space-evenly",
    alignItems: "center",
  },
  text: {
    width: '10%',
    fontStyle: 'italic',
  },
  container_under: {
    height: 50,
  },
  container_first: {
    height: 100,
  },
  container_second: {
    height: 100,
  },
  container_third: {
    height: 100,
  },
});

export default RankingBox