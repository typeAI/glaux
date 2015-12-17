package glaux.persistence.mongodb
import GeneralHandlers._
import GlauxHandlers._
import glaux.reinforcementlearning.{AdvancedQAgent, QAgent, SimpleQAgent}
import reactivemongo.bson.BSONDocument

object QSessionHandler {
  trait Factory[AT <: QAgent] {
    def apply(s: AT): Handler[s.Session]
  }


  implicit val fsimple: Factory[SimpleQAgent] = new Factory[SimpleQAgent] {
    def apply(s: SimpleQAgent): Handler[s.Session] = agentSessionH(s)
  }
  
  implicit val fAdvanced: Factory[AdvancedQAgent] = new Factory[AdvancedQAgent] {
    def apply(s: AdvancedQAgent): Handler[s.Session] = agentSessionH(s)
  }

}
