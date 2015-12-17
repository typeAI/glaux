package glaux.interfaces.akka.api
package domain

import glaux.neuralnetwork.trainers.SGD.SGDSettings
import glaux.reinforcementlearning.DeepMindQLearner.ConvolutionBased

case class SessionId(agentName: AgentName, profileId: ProfileId)

sealed trait AgentSettings {
  def name: AgentName
}

case class AdvancedAgentSettings(
  name:            AgentName,
  numOfActions:    Int,
  learnerSettings: ConvolutionBased.Settings,
  trainerSettings: SGDSettings
) extends AgentSettings
