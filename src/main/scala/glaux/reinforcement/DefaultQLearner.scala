package glaux.reinforcement

import glaux.linalg.{Tensor, RowVector}
import glaux.nn.trainers.{BatchTrainer, MomentumSGD}
import glaux.reinforcement.QLearner._

import scala.util.Random


abstract class DefaultQLearner[NetInput <: Tensor]( historyLength: Int,
                                                    numberOfReadings: Int,
                                                    gamma: Int,
                                                    batchSize: Int,
                                                    override protected val trainer: QLearner[NetInput]#Trainer) extends QLearner[NetInput] {

  def iterate(lastIteration: Iteration, observation: Observation): Iteration = {
    val targetNet = lastIteration.targetNet
    val learningNet = lastIteration.learningNet

    def randomExamples(memory: Memory): Memory = {
      (1 to batchSize).map { _ =>
        memory(Random.nextInt(memory.size))
      }
    }

    def updateMemory: Memory = {
      val after = getState(lastIteration.memory, observation.recentHistory)
      val before = lastIteration.memory.last.after
      val action = observation.lastAction
      lastIteration.memory :+ Transition(before, action, observation.reward, after)
    }


    def trainingExample(example: Transition): (NetInput, NetOutput) = {

      val regressionOnAction = if (example.after.isTerminal) example.reward else
        observation.reward + targetNet.predict(example.after).seqView.max * gamma

      val dataFlow = learningNet.forward(example.before) //todo: eliminate the need to do an extra forward, that is, having trainer be able to train on a single regression diff
      val target = dataFlow.last.out.update(example.action, regressionOnAction)
      (example.before, target)
    }

    val newMemory = updateMemory
    val result = trainer.trainBatch(randomExamples(newMemory).map(trainingExample), lastIteration.trainingResult)

    Iteration(
      targetNet, //todo: periodically update targetNet to learning Net
      newMemory,
      result
    )

  }


  def getState(memory: Memory, recentHistory: History): State = ???


}