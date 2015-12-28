[![Build Status](https://travis-ci.org/typeAI/glaux.svg)](https://travis-ci.org/typeAI/glaux)
[![Codacy Badge](https://api.codacy.com/project/badge/grade/606a8cedadf9441a8f894bc6e1bf22eb)](https://www.codacy.com/app/kailuo-wang/glaux)
[![Codacy Badge](https://api.codacy.com/project/badge/coverage/606a8cedadf9441a8f894bc6e1bf22eb)](https://www.codacy.com/app/kailuo-wang/glaux)
[![Stories in Ready](https://badge.waffle.io/typeAI/glaux.svg?label=ready&title=Ready)](http://waffle.io/typeAI/glaux)


# Glaux - Deep reinforcement learning library in functional scala

## This library is still in an experimental phase - no release yet.


Glaux is an experiment to code in functional scala some deep reinforcement learning algorithms, generally speaking, i.e. deep neural network applied in reinforcement learning. The first algorithm Glaux is set to implement is [DQN](http://www.readcube.com/articles/10.1038%2Fnature14236?shared_access_token=Lo_2hFdW4MuqEcF3CVBZm9RgN0jAjWel9jnR3ZoTv0P5kedCCNjz3FJ2FhQCgXkApOr3ZSsJAldp-tw3IWgTseRnLpAc9xQq-vTA2Z5Ji9lg16_WvCy4SaOgpK5XXA6ecqo8d8J7l4EJsdjwai53GqKt-7JuioG0r3iV67MQIro74l6IxvmcVNKBgOwiMGi8U0izJStLpmQp6Vmi_8Lw_A%3D%3D) by [DeepMind](http://deepmind.com/). 
 

Glaux is modular. As of now glaux consists of 6 modules.  

### linear-algebra
Linear algebra adaptors for easy exchange of underlying linear algebra library for concrete implementation. Right now there is only one implementation based on [nd4j](http://nd4j.org).

### neural-network
A neural network library that is extensible with new types of layers and trainers. 

### reinforcement-learning
Reinformment learning using neural networks for approximate Q functions. 

### interface-api
The API for client usage of the reinforcement learning alorithm defined above 

### akka-interface
An interface application for deep reinforcement learning implemented in AKKA. 

### persistence-mongodb
A persistence library that can persist reinforcement learning sessions into MongoDB. 

Stores agent settings and data into mongodb

## To run tests
For unit tests run
`sbt test` 

For integration tests, start `mongod` and run

`sbt integration:test`

