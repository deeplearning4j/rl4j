# RL4J: Reinforcement Learning for Java

NOTE: RL4J IS NO LONGER SUPPORTED.

RL4J is a reinforcement learning framework integrated with deeplearning4j and released under an Apache 2.0 open-source license. 

* DQN (Deep Q Learning with double DQN)
* Async RL (A3C, Async NStepQlearning)

Both for Low-Dimensional (array of info) and high-dimensional (pixels) input.


![DOOM](docs/images/doom.gif)


![Cartpole](docs/images/cartpole.gif)

A useful blog post to introduce you to reinforcement learning, DQN and Async RL:

[Blog post](https://rubenfiszel.github.io/posts/rl4j/2016-08-24-Reinforcement-Learning-and-DQN.html)

[Examples](https://github.com/eclipse/deeplearning4j-examples/tree/master/rl4j-examples)

[Cartpole example](https://github.com/eclipse/deeplearning4j-examples/blob/master/rl4j-examples/src/main/java/org/deeplearning4j/examples/rl4j/Cartpole.java)

# Quickstart

* mvn install

# Visualisation

[webapp-rl4j](https://github.com/rubenfiszel/webapp-rl4j)

# Quicktry cartpole:

* run with this [main](https://github.com/eclipse/deeplearning4j-examples/blob/master/rl4j-examples/src/main/java/org/deeplearning4j/examples/rl4j/Cartpole.java)

# Doom

Doom is not ready yet but you can make it work if you feel adventurous with some additional steps:

* You will need vizdoom, compile the native lib and move it into the root of your project in a folder
* export MAVEN_OPTS=-Djava.library.path=THEFOLDEROFTHELIB
* mvn compile exec:java -Dexec.mainClass="YOURMAINCLASS"

# Malmo (Minecraft)

![Malmo](docs/images/malmo.gif)

* Download and unzip Malmo from [here](https://github.com/Microsoft/malmo/releases)
* export MALMO_HOME=YOURMALMO_FOLDER
* export MALMO_XSD_PATH=$MALMO_HOME/Schemas
* launch malmo per [instructions](https://github.com/Microsoft/malmo#launching-minecraft-with-our-mod)
* run with this [main](https://github.com/eclipse/deeplearning4j-examples/blob/master/rl4j-examples/src/main/java/org/deeplearning4j/examples/rl4j/MalmoPixels.java)

# WIP

* Documentation
* Serialization/Deserialization (load save)
* Compression of pixels in order to store 1M state in a reasonnable amount of memory
* Async learning: A3C and nstep learning (requires some missing features from dl4j (calc and apply gradients)).

# Author

[Ruben Fiszel](http://rubenfiszel.github.io/) 
