package glaux.persistence.mongodb

import glaux.interfaces.akka.api.domain.{ AgentName, AgentSettings }
import reactivemongo.api.collections.bson.BSONCollection
import reactivemongo.api.indexes.{ IndexType, Index }
import reactivemongo.bson.BSONDocument
import scala.concurrent.ExecutionContext.Implicits.global
import ShifuHandlers._

import scala.concurrent.Future

trait AgentSettingsRepo {

  def get(name: AgentName): Future[Option[AgentSettings]]

  def upsert(settings: AgentSettings): Future[Unit]

}

case class AgentSettingsRepoImpl(collection: BSONCollection) extends AgentSettingsRepo with Repository {

  collection.indexesManager.ensure(Index(Seq(("name", IndexType.Ascending)), name = Some("agentTypeName")))

  def get(name: AgentName): Future[Option[AgentSettings]] = collection.find(BSONDocument("name" → name)).one[AgentSettings]

  def upsert(settings: AgentSettings): Future[Unit] =
    collection.findAndUpdate(
      selector = BSONDocument("name" → settings.name),
      update = settings,
      upsert = true
    )

}

object AgentSettingsRepo {

  def apply: AgentSettingsRepo = {
    AgentSettingsRepoImpl(Repository.collectionOf("session"))
  }
}
