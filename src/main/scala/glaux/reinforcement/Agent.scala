package glaux.reinforcement

trait Agent {
  val qLearner: QLearner
  import qLearner.State
  type Policy = (State, Action => Q) => Action

  val policy: Policy
}
