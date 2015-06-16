package glaux.reinforcement

import glaux.linalg.Dimension.Row
import glaux.linalg.RowVector
import glaux.nn.trainers.{SGDOptions, VanillaSGD}
import glaux.nn.{Net, InputLayer}
import glaux.nn.Net.{Updater, DefaultNet}
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
                               recentHistory: History,
                               inputDimension: Input#Dimensionality,
                               isTerminal: Boolean = false,
                               targetNetHitCount: Int = 0 ) extends IterationLike {
    lazy val net = trainingResult.net
    lazy val latestState = stateFromHistory(recentHistory, isTerminal)
  }

  type Iteration = DeepMindIteration

  val minMemorySizeBeforeTraining: Int

  def stateFromHistory(history: History, isTerminal: Boolean): Option[State] =
    if(history.size >= historyLength) Some(State(history, isTerminal)) else None

  protected def validate: Unit = {
    assert(minMemorySizeBeforeTraining > batchSize, "must have enough transitions in memory before training")
  }

  def init(inputDimension: Input#Dimensionality, numOfActions: Int): Iteration = {
    val initNet = buildNet(inputDimension, numOfActions)
    DeepMindIteration(initNet, Nil, trainer.init(initNet), Nil, inputDimension)
  }

  protected def buildNet(inputDimension: Input#Dimensionality, numOfActions: Int): Net

  def iterate(lastIteration: Iteration, observation: Observation): Iteration = {

    assert(observation.recentHistory.forall(_.readings.dimension == lastIteration.inputDimension), "input readings doesn't conform to preset reading dimension")

    def updateRecentHistory: History = {
      val relevantPreviousHistory = lastIteration.recentHistory.filter(_.time.isBefore(observation.startTime)) //remove duplicated temporal state when there is an overlap between two observations
      (relevantPreviousHistory ++ observation.recentHistory).takeRight(historyLength)
    }


    def updateMemory(recentHistory: History): Option[Memory] = {
      for {
        before <- if(lastIteration.isTerminal) None else lastIteration.latestState
        after <- stateFromHistory(recentHistory, observation.isTerminal)
      } yield
        lastIteration.memory :+ Transition(before, observation.lastAction, observation.reward, after)
    }

    val recentHistory = updateRecentHistory
    val newMemory = updateMemory(recentHistory).getOrElse(lastIteration.memory)
    val doTraining: Boolean = newMemory.size > minMemorySizeBeforeTraining
    val updateTarget: Boolean = doTraining && lastIteration.targetNetHitCount > targetNetUpdateFreq

    val targetNet = lastIteration.targetNet
    lazy val newResult = train(newMemory, lastIteration.trainingResult, targetNet)

    lastIteration.copy(
      targetNet = if(updateTarget) newResult.net else targetNet,
      memory = newMemory,
      trainingResult =  if(doTraining) newResult else lastIteration.trainingResult,
      recentHistory = recentHistory,
      isTerminal = observation.isTerminal,
      targetNetHitCount = if(updateTarget) 0 else lastIteration.targetNetHitCount + 1
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
  case class Simplified(  override protected val trainer: Simplified#Trainer = VanillaSGD[DeepMindQLearner.Simplified#Net](SGDOptions()),
                          historyLength: Int = 50,
                          gamma: Double = 0.95,
                          batchSize: Int = 40,
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
      val inputSize = inputDimension.size * historyLength
      val netInputDimension = Row(inputSize)
      val inputLayer = InputLayer[RowVector](netInputDimension)
      val fc1 = FullyConnected(inputSize, inputSize)
      val relu = Relu[RowVector](netInputDimension)
      val fc2 = FullyConnected(inputSize, numOfActions)
      val lossLayer = Regression(numOfActions)
      DefaultNet(inputLayer, Seq(fc1, relu, fc2), lossLayer)
    }
  }

}