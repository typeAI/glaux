package glaux.persistence.mongodb

import glaux.interfaces.api.domain.{AgentName, AgentSettings}
import glaux.interfaces.api.persistence.{AgentSettingsPersistence, Persistence}
import reactivemongo.api.collections.bson.BSONCollection
import reactivemongo.api.indexes.{IndexType, Index}
import reactivemongo.bson.BSONDocument
import scala.concurrent.ExecutionContext.Implicits.global
import InterfaceHandlers._

import scala.concurrent.Future

case class AgentSettingsRepoImpl(collection: BSONCollection) extends Persistence[AgentSettings, AgentName] with Repository {

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

  def apply(): AgentSettingsPersistence = {
    AgentSettingsRepoImpl(Repository.collectionOf("session"))
  }
}
