package glaux.reinforcement

import glaux.linalg.Dimension.Row
import glaux.linalg.RowVector
import glaux.nn.{Net, InputLayer}
import glaux.nn.Net.{CanBuildFrom, DefaultNet}
import glaux.nn.layers.{Regression, Relu, FullyConnected}

import scala.util.Random

/**
 * QLearner based on deepmind algorithm
 */
trait DeepMindQLearner extends QLearner {
  val historyLength: Int
  val gamma: Double
  val batchSize: Int
  val targetNetUpdateFreq: Int //avg # of iterations before updating the target net
  case class DeepMindIteration(targetNet: Net,
                       memory: Memory, //Seq because we need random access here
                       trainingResult: TrainingResult,
                       targetNetHitCount: Int = 0 ) extends IterationLike {
    lazy val net = trainingResult.net
  }

  type Iteration = DeepMindIteration

  val minMemorySizeBeforeTraining: Int

  protected def validate: Unit = {
    assert(minMemorySizeBeforeTraining > batchSize, "must have enough transitions in memory before training")
  }

  def init(inputDimension: Input#Dimensionality, numOfActions: Int): Iteration = {
    val initNet = buildNet(inputDimension, numOfActions)
    DeepMindIteration(initNet, Nil, trainer.init(initNet))
  }

  protected def buildNet(inputDimension: Input#Dimensionality, numOfActions: Int): Net

  def iterate(lastIteration: Iteration, observation: Observation): Iteration = {

    assert(observation.recentHistory.forall(_.readings.dimension == lastIteration.inputDimension), "input readings doesn't conform to preset reading dimension")

    def updateMemory: Memory = {
      val before = lastIteration.latestState

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

    DeepMindIteration(
      if(updateTarget) newResult.net else targetNet,
      newMemory,
      if(doTraining) newResult else lastIteration.trainingResult,
      if(updateTarget) 0 else lastIteration.targetNetHitCount + 1
    )

  }

  private def train(memory: Memory, lastResult: TrainingResult, targetNet: Net): TrainingResult = {
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
  case class Simplified(  override protected val trainer: Simplified#Trainer,
                          historyLength: Int = 50,
                          gamma: Double = 0.95,
                          batchSize: Int = 32,
                          targetNetUpdateFreq: Int = 10, //avg # of iterations before updating the target net
                          minMemorySizeBeforeTraining: Int = 100 ) extends DeepMindQLearner {
    type NetInput = RowVector
    type Input = RowVector
    type Net = DefaultNet[NetInput]
    validate

    implicit def inputToNet(state: State): NetOutput = {
      RowVector(state.fullHistory.flatMap(_.readings.seqView):_*)
    }

    protected def buildNet(inputDimension: Input#Dimensionality, numOfActions: Int): Net = {
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