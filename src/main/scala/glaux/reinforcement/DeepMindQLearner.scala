package glaux.reinforcement

import glaux.linalg.Dimension.{ThreeD, Row}
import glaux.linalg.{Vol, RowVector}
import glaux.nn.trainers.VanillaSGD
import glaux.nn.trainers.SGD.SGDSettings
import glaux.nn.{Rectangle, InputLayer, Net}
import Net.DefaultNet
import glaux.nn.layers.{Convolution, Regression, Relu, FullyConnected}
import glaux.reinforcement.QLearner.Transition

import scala.util.Random

/**
 * QLearner based on deepmind algorithm
 */
trait DeepMindQLearner extends QLearner {
  val gamma: Double
  val batchSize: Int
  val targetNetUpdateFreq: Int //avg # of iterations before updating the target net
  type Net = DefaultNet[NetInput]
  type Trainer = VanillaSGD[Net]
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



  override protected def doInit(initState: State, numOfActions: Action, inputDim: InputDimension): Iteration = {
    val initNet = buildNet(inputDim, numOfActions)

    DeepMindIteration(initNet, Nil, trainer.init(initNet), initState)
  }

  override protected def doIterate(lastIteration: Iteration, observation: Observation, currentState: State): Iteration = {
    val newMemory = if (lastIteration.isTerminal) lastIteration.memory
      else {
        val before = lastIteration.state
        lastIteration.memory :+ Transition(before, observation.lastAction, observation.reward, currentState)
      }

    val doTraining: Boolean = newMemory.size > minMemorySizeBeforeTraining
    val updateTarget: Boolean = doTraining && lastIteration.targetNetHitCount > targetNetUpdateFreq

    val targetNet = lastIteration.targetNet
    lazy val newResult = train(newMemory, lastIteration.trainingResult, targetNet)

    lastIteration.copy(
      targetNet = if (updateTarget) newResult.net else targetNet,
      memory = newMemory,
      trainingResult = if (doTraining) newResult else lastIteration.trainingResult,
      state = currentState,
      isTerminal = observation.isTerminal,
      targetNetHitCount = if (updateTarget) 0 else lastIteration.targetNetHitCount + 1
    )
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

    def toTrainingInput(transition: Transition[Input]): (NetInput, Trainer#ScalarOutputInfo) = {
      val regressionOnAction = if (transition.after.isTerminal) transition.reward else
        transition.reward + targetNet.predict(transition.after).seqView.max * gamma

      (transition.before, (regressionOnAction, transition.action))
    }
    trainer.trainBatchWithScalaOutputInfo(lastResult)(randomExamples.map(toTrainingInput))
  }

}

object DeepMindQLearner {
  case class Simplified(  override protected val trainer: Simplified#Trainer = VanillaSGD[Simplified#Net](SGDSettings()),
                          historyLength: Int = 50,
                          gamma: Double = 0.95,
                          batchSize: Int = 40,
                          targetNetUpdateFreq: Int = 10, //avg # of iterations before updating the target net
                          minMemorySizeBeforeTraining: Int = 100 ) extends DeepMindQLearner {
    type NetInput = RowVector
    type Input = RowVector


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

  case class ConvolutionBased(override protected val trainer: ConvolutionBased#Trainer = VanillaSGD[ConvolutionBased#Net](SGDSettings()),
                              historyLength: Int = 50,
                              filterSize: Int = 5,
                              gamma: Double = 0.95,
                              batchSize: Int = 20,
                              targetNetUpdateFreq: Int = 10, //avg # of iterations before updating the target net
                              minMemorySizeBeforeTraining: Int = 100 ) extends DeepMindQLearner {
    type NetInput = Vol
    type Input = RowVector


    validate
    assert(historyLength > filterSize * 2, "too short history makes convolution useless")

    implicit def inputToNet(state: State): NetInput = {
      Vol(netInputDimension(state.fullHistory.head.readings.dimension), state.fullHistory.flatMap(_.readings.seqView))
    }


    protected def buildNet(inputDimension: Input#Dimensionality, numOfActions: Int): Net = {
      val netInputDim = netInputDimension(inputDimension)
      val inputLayer = InputLayer[Vol](netInputDim)
      val conv1 = Convolution(
        numOfFilters = 10,
        filterSize = Rectangle(filterSize, 1),
        inputDimension = netInputDim,
        padding = true
      )
      val relu1 = Relu[Vol](conv1.outDimension)
      val conv2 = Convolution(
        numOfFilters = 10,
        filterSize = Rectangle(filterSize, 1),
        inputDimension = conv1.outDimension,
        padding = true
      )
      val relu2 = Relu[Vol](conv2.outDimension)
      val fc1 = FullyConnected[Vol](conv2.outDimension, 30)
      val fc2 = FullyConnected(30, numOfActions)
      val lossLayer = Regression(numOfActions)
      DefaultNet(inputLayer, Seq(conv1, relu1, conv2, relu2, fc1, fc2), lossLayer)
    }

    private def netInputDimension(inputDimension: Row): ThreeD = ThreeD(historyLength, 1, inputDimension.size)
  }

}
