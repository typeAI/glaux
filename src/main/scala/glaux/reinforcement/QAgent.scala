package glaux.reinforcement

import glaux.linalg.Dimension.Row
import glaux.linalg.Tensor.TensorBuilder
import glaux.nn.trainers.{SGDOptions, VanillaSGD}
import glaux.reinforcement.DeepMindQLearner.Simplified

import scala.util.Random

trait QAgent {
  val qLearner: QLearner

  implicit val builder : TensorBuilder[qLearner.Input]

  type Iteration = qLearner.Iteration

  import qLearner.Observation
  type Policy = (Iteration) => Action

  import qLearner.TemporalState

  def init(initReadings: Iterable[Reading]): Either[String, Iteration]
  
  def updateInitIteration(iteration: Iteration, newReading: Reading): Iteration =
    qLearner.updateInit(iteration, toHistory(Seq(newReading), iteration.state.inputDimension))

  def iterate(lastIteration: Iteration, lastAction: Action, reward: Reward, readings: Iterable[Reading]): (Iteration, Action) = {
    val observation = Observation(lastAction, reward, toHistory(readings, lastIteration.state.inputDimension), false)
    val newIteration = qLearner.iterate(lastIteration, observation)
    (newIteration, policy(newIteration))
  }

  val policy: Policy

  protected def toHistory(readings: Iterable[Reading], dimension: qLearner.InputDimension): qLearner.History = {
    readings.toSeq.map {
      case (r, time) => TemporalState(builder((dimension, r.toSeq)), time)
    }
  }
}


class SimpleQAgent(numOfReads: Int, numOfActions: Int) extends QAgent {
  val trainer = VanillaSGD[Simplified#Net](SGDOptions(learningRate = 0.05))
  val qLearner = DeepMindQLearner.Simplified(historyLength = 10, batchSize = 20, trainer = trainer)

  implicit val builder = implicitly[TensorBuilder[qLearner.Input]]

  def init(initReadings: Iterable[Reading]) = qLearner.init(toHistory(initReadings, Row(initReadings.head._1.size)), numOfActions)

  val policy = (iteration: Iteration) => Random.nextInt(numOfActions) //todo: implement a real policy
}