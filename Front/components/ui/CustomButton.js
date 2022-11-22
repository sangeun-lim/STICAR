import React, { Component } from 'react';
import { TouchableOpacity, Text, StyleSheet } from 'react-native';

const CustomButton = ({ onPress, text, type = "PRIMARY" }) => {
    return (
        <TouchableOpacity
            onPress={onPress}
            style={[styles.button, styles[`button_${type}`]]} >
            <Text style={[styles.text, styles[`text_${type}`]]}>{text}</Text>
        </TouchableOpacity>
    );
};

const styles = StyleSheet.create({
    button: {
        width: '90%',
        height: '30%',
        maxHeight: 50,
        borderRadius: 10,
        marginVertical: 10,
        borderWidth: 0,
        alignItems: 'center',
        justifyContent: 'center',
    },
    button_PRIMARY: {
        backgroundColor: '#3B71F3'
    },
    button_SignUp: {
        backgroundColor: 'yellow',
    },
    button_Login: {
        backgroundColor: '#289BE4',
    },
    button_FindPassword: {
    },

    text: {
        color: 'black',
        fontWeight: 'bold',
        fontSize: 16,
    },
    text_FindPassword: {
        color: 'gray',
    },
});

export default CustomButton;