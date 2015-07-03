package glaux.reinforcement

import glaux.linalg.Dimension.Row
import glaux.linalg.RowVector
import glaux.nn.trainers.{SGDOptions, VanillaSGD}
import glaux.nn.{Net, InputLayer}
import glaux.nn.Net.{Updater, DefaultNet}
import glaux.nn.layers.{Regression, Relu, FullyConnected}

import scala.util.{Try, Random}

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
                               state: State,
                               isTerminal: Boolean = false,
                               targetNetHitCount: Int = 0 ) extends IterationLike {
    lazy val net = trainingResult.net

  }

  assert(historyLength > 0)

  type Iteration = DeepMindIteration

  val minMemorySizeBeforeTraining: Int

  protected def buildNet(inputDimension: Input#Dimensionality, numOfActions: Int): Net

  def init(initHistory: History, numOfActions: Int): Either[String, Iteration] = {
    if (initHistory.length < historyLength) {
      Left("not enough history")
    } else {
      val inputDim = inputDimensionOfHistory(initHistory)
      if(inputDim.isEmpty)
         Left("readings doens't have consistent dimension")
      else {
        val initNet = buildNet(inputDim.get, numOfActions)
        Right(DeepMindIteration(initNet, Nil, trainer.init(initNet), stateFromHistory(initHistory, false)))
      }
    }
  }

  def iterate(lastIteration: Iteration, observation: Observation): Iteration = {

    assert(observation.recentHistory.forall(_.readings.dimension == lastIteration.state.inputDimension),
      s"input readings doesn't conform to preset reading dimension ${lastIteration.state.inputDimension}")

    val relevantHistory = if(lastIteration.isTerminal) observation.recentHistory else concat(lastIteration.state.fullHistory, observation.recentHistory)

    val currentState = stateFromHistory(relevantHistory, observation.isTerminal)

    val newMemory = if (lastIteration.isTerminal) lastIteration.memory else {
        val before = lastIteration.state
        lastIteration.memory :+ Transition(before, observation.lastAction, observation.reward, currentState)
      }

    val doTraining: Boolean = newMemory.size > minMemorySizeBeforeTraining
    val updateTarget: Boolean = doTraining && lastIteration.targetNetHitCount > targetNetUpdateFreq

    val targetNet = lastIteration.targetNet
    lazy val newResult = train(newMemory, lastIteration.trainingResult, targetNet)

    lastIteration.copy(
      targetNet = if(updateTarget) newResult.net else targetNet,
      memory = newMemory,
      trainingResult =  if(doTraining) newResult else lastIteration.trainingResult,
      state = currentState,
      isTerminal = observation.isTerminal,
      targetNetHitCount = if(updateTarget) 0 else lastIteration.targetNetHitCount + 1
    )

  }

  private def concat(previous: History, newHistory: History): History = {
    val relevantPreviousHistory = previous.filter(_.time.isBefore(newHistory.head.time))
    (relevantPreviousHistory ++ newHistory)
  }


  private[reinforcement] def updateInit(iteration: Iteration, newHistory: History): Iteration = {
    val previous = iteration.state.fullHistory
    assert(!iteration.state.isTerminal, "init state cannot be terminal")
    assert(iteration.memory.isEmpty, "init iteration should not have transition memory already")
    iteration.copy(state = stateFromHistory(concat(previous, newHistory) , false))
  }



  protected def stateFromHistory(history: History, isTerminal: Boolean): State = {
    assert(history.size >= historyLength, "incorrect history length to create a state")
    assert(inputDimensionOfHistory(history).isDefined, "Inconsistent history input dimension")
    State(history.takeRight(historyLength), isTerminal)
  }


  protected def validate: Unit = {
    assert(minMemorySizeBeforeTraining > batchSize, "must have enough transitions in memory before training")
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
  case class Simplified(  override protected val trainer: Simplified#Trainer = VanillaSGD[Simplified#Net](SGDOptions()),
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