package glaux.reinforcement

import glaux.linalg.Tensor
import glaux.nn.trainers.BatchTrainer
import glaux.reinforcement.QLearner._

import scala.util.Random


abstract class DefaultQLearner[NetInput <: Tensor]( historyLength: Int = 50,
                                                    numberOfReadings: Int,
                                                    gamma: Int,
                                                    batchSize: Int = 32,
                                                    targetNetUpdateFreq: Int = 10, //avg # of iterations before updating the target net
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
      val after = produceState(lastIteration.memory, observation.recentHistory)
      val before = lastIteration.memory.last.after
      val action = observation.lastAction
      lastIteration.memory :+ Transition(before, action, observation.reward, after)
    }

    def trainingInput(transition: Transition): (NetInput, Trainer#ScalarOutputInfo) = {
      val regressionOnAction = if (transition.after.isTerminal) transition.reward else
        transition.reward + targetNet.predict(transition.after).seqView.max * gamma

      (transition.before, (regressionOnAction, transition.action))
    }

    val newMemory = updateMemory
    val result = trainer.trainBatchWithScalaOutputInfo(randomExamples(newMemory).map(trainingInput), lastIteration.trainingResult)
    val updateTarget = lastIteration.targetNetHitCount > targetNetUpdateFreq

    Iteration(
      if(updateTarget) result.net else targetNet,
      newMemory,
      result,
      if(updateTarget) 0 else lastIteration.targetNetHitCount + 1
    )

  }

  def produceState(memory: Memory, recentHistory: History): State = ???


}