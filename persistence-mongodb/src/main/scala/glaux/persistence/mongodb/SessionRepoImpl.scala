package glaux.persistence.mongodb

import glaux.interfaces.api.domain.SessionId
import glaux.interfaces.api.persistence.SessionPersistence
import glaux.persistence.mongodb.GeneralHandlers.Handler
import glaux.reinforcementlearning.QAgent
import reactivemongo.api.collections.bson.BSONCollection
import reactivemongo.api.commands.WriteResult
import reactivemongo.api.indexes.{IndexType, Index}
import reactivemongo.bson.{BSONDocument, Macros}
import sun.management.resources.agent

import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global
import InterfaceHandlers._

trait QAgentSessionPersistence[A <: QAgent] extends SessionPersistence[A] {

  implicit def sessionHandler(implicit agent: A): Handler[agent.Session]

  def repo(implicit agent: A) = SessionRepoImpl(agent) //todo: this is obviously not performance friendly, use AUX to solve the dependent type problem will help solve this design.   

  def get(agent: A, id: SessionId): Future[Option[agent.Session]] = repo(agent).get(id)

  def upsert(agent: A, id: SessionId)(session: agent.Session): Future[Unit] = repo(agent).upsert(id, session)
}

case class SessionRepoImpl[Session <: QAgent.Session[_, _]: Handler](collection: BSONCollection) extends Repository {

  collection.indexesManager.ensure(Index(Seq(
    ("id.profileId", IndexType.Ascending),
    ("id.agentName", IndexType.Ascending)), name = Some("agentId_index")))

  private case class SessionRecord(id: SessionId, session: Session)

  implicit private val srf = Macros.handler[SessionRecord]

  def get(id: SessionId): Future[Option[Session]] =
    collection.find(idSelector(id)).one[SessionRecord].map(_.map(_.session))

  def insert(sessionId: SessionId, session: Session): Future[Unit] = collection.insert(SessionRecord(sessionId, session))

  def upsert(id: SessionId, session: Session): Future[Unit] = upsert(idSelector(id), SessionRecord(id, session))

  private def idSelector(id: SessionId) : BSONDocument = BSONDocument("id.profileId" -> id.profileId, "id.agentName" -> id.agentName )

}


object SessionRepoImpl {
  def apply[AT <: QAgent](qAgent: AT)(implicit h: Handler[qAgent.Session]): SessionRepoImpl[qAgent.Session] = {
    SessionRepoImpl[qAgent.Session](collection)
  }

  private[glaux] def removeAll(): Future[WriteResult] = collection.remove(BSONDocument.empty)

  private lazy val collection: BSONCollection = Repository.collectionOf("session")

  def close() = collection.db.connection.close()

}
