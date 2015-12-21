package glaux.persistence.mongodb

import glaux.interfaces.api.domain.SessionId
import glaux.interfaces.api.persistence.SessionPersistence
import glaux.persistence.mongodb.GeneralHandlers.Handler
import glaux.reinforcementlearning.QAgent
import reactivemongo.api.collections.bson.BSONCollection
import reactivemongo.api.commands.WriteResult
import reactivemongo.api.indexes.{IndexType, Index}
import reactivemongo.bson.BSONDocument

import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global
import InterfaceHandlers._
import shapeless.cachedImplicit

trait QAgentSessionPersistence[A <: QAgent] extends SessionPersistence[A] {

  implicit def sessionHandler(implicit agent: A): Handler[agent.Session]

  def repo(implicit agent: A) = SessionRepoImpl(agent) //todo: creating an instance every time is not performance friendly - need to bench mark

  def get(agent: A, id: SessionId): Future[Option[agent.Session]] = repo(agent).get(id)

  def upsert(agent: A, id: SessionId)(session: agent.Session): Future[Unit] = repo(agent).upsert(id, session)
}

case class SessionRepoImpl[Session <: QAgent.Session[_, _]: Handler](collection: BSONCollection) extends Repository {
  implicit private val handler: Handler[SessionRecord[Session]] = cachedImplicit

  def get(id: SessionId): Future[Option[Session]] =
    collection.find(idSelector(id)).one[SessionRecord[Session]].map(_.map(_.session))

  def insert(sessionId: SessionId, session: Session): Future[Unit] = collection.insert(SessionRecord(sessionId, session))

  def upsert(id: SessionId, session: Session): Future[Unit] = upsert(idSelector(id), SessionRecord(id, session))

  private def idSelector(id: SessionId) : BSONDocument = BSONDocument("id.profileId" -> id.profileId, "id.agentName" -> id.agentName )

}

private[mongodb] case class SessionRecord[Session <: QAgent.Session[_, _]: Handler](id: SessionId, session: Session)


object SessionRepoImpl {

  def apply[AT <: QAgent](qAgent: AT)(implicit h: Handler[qAgent.Session]): SessionRepoImpl[qAgent.Session] = {
    SessionRepoImpl[qAgent.Session](collection)
  }

  private[glaux] def removeAll(): Future[WriteResult] = collection.remove(BSONDocument.empty)

  private lazy val collection: BSONCollection = {
    val col = Repository.collectionOf("session")
    col.indexesManager.ensure(Index(Seq(
      ("id.profileId", IndexType.Ascending),
      ("id.agentName", IndexType.Ascending)), name = Some("agentId_index")))
    col
  }

  def close() = collection.db.connection.close()

}
