import React, { Component } from 'react'
import { View, Text, TextInput, StyleSheet } from 'react-native'
import Icon from 'react-native-vector-icons/FontAwesome'

class CustomInput extends Component {
  state = { isFocused: false };

  onFocusChange = () => {
    this.setState({ isFocused: true });
  }

  render() {
    return (
      <View style={[styles.container, { borderColor: this.stateisFocused ? '#0779ef' : '#eee' }]}>
        <Icon
          style={styles.icon}
          name={this.props.icon}
          size={22}
          color={!this.state.isFocused ? 'gray' : (
            this.props.valueBoolean ? 'tomato' : '#A7D93D')} />
        <TextInput
          placeholder={this.props.placeholder}
          onFocus={this.onFocusChange}
          style={styles.textinputcontainer}
          secureTextEntry={this.props.secureTextEntry}

          value={this.props.value}
          onChangeText={this.props.setValue} />
      </View>
    );
  };
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    width: '90%',
    height: 55,
    borderRadius: 10,
    marginVertical: 10,
    borderWidth: 3.5,
    alignItems: 'center',
    backgroundColor: '#F7F3E9',
  },
  textinputcontainer: {
    width: '100%',
    borderBottomWidth: 0,
    color: '#404040',
    fontWeight: 'bold',
    marginLeft: 5,
  },
  icon: {
    marginHorizontal: 10,
  }
})

export default CustomInput;