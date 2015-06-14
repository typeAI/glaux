package glaux.reinforcement

import glaux.linalg.Dimension.{TwoD, Row}
import glaux.linalg.{Matrix, Dimension, RowVector, Tensor}
import glaux.nn.{HiddenLayer, InputLayer}
import glaux.nn.Net.DefaultNet
import glaux.nn.layers.{Regression, Softmax, Relu, FullyConnected}

import scala.util.Random

/**
 * QLearner based on deepmind algorithm
 */
trait DeepMindQLearner extends QLearner {
  val historyLength: Int
  val inputDimension: Input#Dimensionality
  val gamma: Int
  val batchSize: Int
  val targetNetUpdateFreq: Int //avg # of iterations before updating the target net
  case class MyIteration(targetNet: Net,
                       memory: Memory, //Seq because we need random access here
                       trainingResult: TrainerResult,
                       targetNetHitCount: Int = 0 ) extends GenIteration {
    lazy val net = trainingResult.net
  }

  type Iteration = MyIteration

  val minMemorySizeBeforeTraining: Int

  protected def validate: Unit = {
    assert(minMemorySizeBeforeTraining > batchSize, "must have enough transitions in memory before training")
  }

  def init(inputDimension: Input#Dimensionality, numOfActions: Int): Iteration = {
    val initNet = buildNet(inputDimension, numOfActions)
    MyIteration(initNet, Nil, trainer.init(initNet))
  }

  def buildNet(inputDimension: Input#Dimensionality, numOfActions: Int): Net

  def iterate(lastIteration: Iteration, observation: Observation): Iteration = {

    assert(observation.recentHistory.forall(_.readings.dimension == inputDimension), "input readings doesn't conform to preset reading dimension")

    def updateMemory: Memory = {
      val before = lastIteration.memory.last.after

      val relevantPreviousHistory = before.fullHistory.filter(_.time.isBefore(observation.recentHistory.head.time))
      val after = State((relevantPreviousHistory ++ observation.recentHistory).takeRight(historyLength), observation.isTerminal)

      val action = observation.lastAction
      lastIteration.memory :+ Transition(before, action, observation.reward, after)
    }

    val newMemory = updateMemory
    val doTraining: Boolean = newMemory.size > minMemorySizeBeforeTraining
    val updateTarget: Boolean = doTraining && lastIteration.targetNetHitCount > targetNetUpdateFreq

    val targetNet = lastIteration.targetNet
    lazy val newResult = train(newMemory, lastIteration.trainingResult, targetNet)

    MyIteration(
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

    trainer.trainBatchWithScalaOutputInfo(lastResult)(randomExamples.map(toTrainingInput))
  }

}

object DeepMindQLearner {
  case class Simplified(  historyLength: Int = 50,
                          inputDimension: Row,
                          gamma: Int,
                          batchSize: Int = 32,
                          targetNetUpdateFreq: Int = 10, //avg # of iterations before updating the target net
                          override protected val trainer: Simplified#Trainer,
                          minMemorySizeBeforeTraining: Int = 100 ) extends DeepMindQLearner {
    type NetInput = RowVector
    type Input = RowVector


    validate

    implicit def toInput(state: State): NetOutput = {
      RowVector(inputDimension, state.fullHistory.flatMap(_.readings.seqView))
    }

    def buildNet(inputDimension: Input#Dimensionality, numOfActions: Int): Net = {
      val inputSize = inputDimension.size
      val inputLayer = InputLayer[RowVector](inputDimension)
      val fc1 = FullyConnected(inputSize, inputSize)
      val relu = Relu[RowVector](inputDimension)
      val fc2 = FullyConnected(inputSize, numOfActions)
      val lossLayer = Regression(numOfActions)
      DefaultNet(inputLayer, Seq(fc1, relu, fc2), lossLayer)

    }
  }


}