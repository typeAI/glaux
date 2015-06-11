package glaux.reinforcement

import glaux.reinforcement.QLearner._


case class DefaultQLearner(historyLength: Int,
                           numberOfReadings: Int,
                           gamma: Int,
                           batchSize: Int) extends QLearner {
  def iterate(lastIteration: Iteration, observation: Observation): Iteration = ???
}