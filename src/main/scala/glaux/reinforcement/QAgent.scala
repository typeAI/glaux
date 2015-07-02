package glaux.reinforcement

import glaux.linalg.Dimension.Row
import glaux.nn.trainers.{SGDOptions, VanillaSGD}
import glaux.reinforcement.DeepMindQLearner.Simplified

import scala.util.Random

trait QAgent {
  val qLearner: QLearner

  type State = qLearner.State
  type Iteration = qLearner.Iteration
  type History = qLearner.History

  type Policy = (Iteration) => Action

  def init(initHistory: History): Either[String, Iteration]
  val policy: Policy
}


class SimpleQAgent(numOfReads: Int, numOfActions: Int) extends QAgent {
  val trainer = VanillaSGD[Simplified#Net](SGDOptions(learningRate = 0.05))
  val qLearner = DeepMindQLearner.Simplified(historyLength = 10, batchSize = 20, trainer = trainer)

  def init(initHistory: History) = qLearner.init(initHistory, numOfActions)

  val policy = (iteration: Iteration) => Random.nextInt(numOfActions) //todo: implement a real policy
}