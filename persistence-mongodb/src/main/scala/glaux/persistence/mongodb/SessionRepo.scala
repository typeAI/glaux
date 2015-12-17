package glaux.persistence.mongodb

import glaux.interfaces.akka.api.domain.SessionId
import glaux.persistence.mongodb.GeneralHandlers.Handler
import glaux.reinforcementlearning.QAgent
import reactivemongo.api.collections.bson.BSONCollection
import reactivemongo.api.commands.WriteResult
import reactivemongo.api.indexes.{IndexType, Index}
import reactivemongo.bson.{BSONDocument, Macros}

import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global
import ShifuHandlers._

case class SessionRepo[Session <: QAgent.Session[_, _]: Handler](collection: BSONCollection) extends Repository {

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


object SessionRepo {


  def apply[AT <: QAgent](qAgent: AT)(implicit f: QSessionHandler.Factory[AT]): SessionRepo[qAgent.Session] = {
    implicit val handler = f(qAgent)
    SessionRepo[qAgent.Session](collection)
  }

  private[glaux] def removeAll(): Future[WriteResult] = collection.remove(BSONDocument.empty)

  private lazy val collection: BSONCollection = Repository.collectionOf("session")

  def close() = collection.db.connection.close()

}
