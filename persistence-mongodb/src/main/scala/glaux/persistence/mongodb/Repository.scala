package glaux.persistence.mongodb

import glaux.interfaces.akka.api.domain._
import glaux.persistence.mongodb.GeneralHandlers.Handler
import reactivemongo.api.MongoDriver
import reactivemongo.api.collections.bson.BSONCollection
import reactivemongo.api.commands.WriteResult
import reactivemongo.bson.BSONDocument

import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global

trait Repository {
  val collection: BSONCollection
  import collection.BatchCommands.FindAndModifyCommand.FindAndModifyResult
  import collection.BatchCommands.FindAndModifyCommand.UpdateLastError
  implicit protected def writeResultToUnit(r: Future[WriteResult]): Future[Unit] = r.collect {
    case wr: WriteResult if wr.ok ⇒ ()
  }

  implicit protected def modifyResultToUnit(r: Future[FindAndModifyResult]): Future[Unit] = r.flatMap {
    case FindAndModifyResult(Some(le), _) if le.err.isEmpty ⇒ Future.successful(())
    case r: FindAndModifyResult                             ⇒ Future.failed(new FailedUpdate(r.lastError.get))
  }

  case class FailedUpdate(le: UpdateLastError) extends Exception(le.err.toString)

  def upsert[T: Handler](idSelector: BSONDocument, item: T): Future[Unit] =
    collection.findAndUpdate(selector = idSelector, update = item, upsert = true)

}

object Repository {
  def collectionOf(name: String): BSONCollection = {
    val driver = new MongoDriver
    val connection = driver.connection(List("localhost"))
    val db = connection("shifu")
    db(name)
  }
}
