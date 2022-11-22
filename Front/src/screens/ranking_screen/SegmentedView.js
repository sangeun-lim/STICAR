import React, { Component } from 'react'
import { StyleSheet, Text, View } from 'react-native'
import SegmentedControlTab from 'react-native-segmented-control-tab'

import DailyRanking from './DailyRanking'
import WeeklyRanking from './WeeklyRanking'
import MonthlyRanking from './MonthlyRanking'

class SegmentedView extends Component {
  constructor() {
    super()
    this.state = {
      selectedIndex: 0,
    }
  }

  handleSingleIndexSelect = (index) => {
    this.setState(prevState => ({ ...prevState, selectedIndex: index }))
  }

  render() {
    const { selectedIndex } = this.state
    return (
      <View style={styles.container}>
        <SegmentedControlTab
          values={["Daily", "Weekly", "Monthly"]}
          selectedIndex={selectedIndex}
          tabStyle={styles.tabStyle}
          activeTabStyle={styles.activeTabStyle}
          onTabPress={this.handleSingleIndexSelect}
        />
        {selectedIndex === 0
          && <DailyRanking />}
        {selectedIndex === 1
          && <WeeklyRanking />}
        {selectedIndex === 2
          && <MonthlyRanking />}
      </View>
    )
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'white',
    padding: 10,
  },
  tabContent: {
    color: '#444444',
    fontSize: 18,
    margin: 24,
  },
  tabStyle: {
    // borderColor: '#D52C43',
  },
  activeTabStyle: {
    // backgroundColor: '#D52C43',
  },
});

export default SegmentedView