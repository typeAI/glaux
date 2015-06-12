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
                                                    override protected val trainer: QLearner[NetInput]#Trainer,
                                                    minMemorySizeBeforeTraining: Int = 100
                                                    ) extends QLearner[NetInput] {
  assert(minMemorySizeBeforeTraining > batchSize, "must have enough transitions in memory before training")

  def iterate(lastIteration: Iteration, observation: Observation): Iteration = {
    val targetNet = lastIteration.targetNet

    def updateMemory: Memory = {
      val after = produceState(lastIteration.memory, observation.recentHistory)
      val before = lastIteration.memory.last.after
      val action = observation.lastAction
      lastIteration.memory :+ Transition(before, action, observation.reward, after)
    }

    val newMemory = updateMemory
    val doTraining: Boolean = newMemory.size > minMemorySizeBeforeTraining
    val updateTarget: Boolean = doTraining && lastIteration.targetNetHitCount > targetNetUpdateFreq

    lazy val newResult = train(newMemory, lastIteration.trainingResult, targetNet)

    Iteration(
      if(updateTarget) newResult.net else targetNet,
      newMemory,
      if(doTraining) newResult else lastIteration.trainingResult,
      if(updateTarget) 0 else lastIteration.targetNetHitCount + 1
    )

  }

  private def train(memory: Memory, lastResult: TrainerResult, targetNet: Net): TrainerResult = {
    def randomExamples: Memory = {
      (1 to batchSize).map { _ =>
        memory(Random.nextInt(memory.size))
      }
    }

    def toTrainingInput(transition: Transition): (NetInput, Trainer#ScalarOutputInfo) = {
      val regressionOnAction = if (transition.after.isTerminal) transition.reward else
        transition.reward + targetNet.predict(transition.after).seqView.max * gamma

      (transition.before, (regressionOnAction, transition.action))
    }

    trainer.trainBatchWithScalaOutputInfo(randomExamples.map(toTrainingInput), lastResult)
  }

  def produceState(memory: Memory, recentHistory: History): State = ???


}