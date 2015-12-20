package glaux.persistence.mongodb

import java.time.ZonedDateTime

import glaux.interfaces.api.domain._
import glaux.persistence.mongodb.GeneralHandlers._
import org.specs2.mutable.Specification
import reactivemongo.bson._
import play.api.libs.functional._
import syntax._

case class TestCase(readings: Vector[(Seq[Double], ZonedDateTime)])
case class TestCase2(readings: Seq[Boolean])

class GeneralFormatsSpec extends Specification {

  def canHandle[T: Handler](t: T) =
    BSON.write(t).as[T] === t

  "Formats" should {

    "format Tuple with option and vectors" in {
      implicit val f = Macros.handler[TestCase]
      canHandle(TestCase(Vector((Seq(12d, 2d), ZonedDateTime.now))))
    }

    "format with serialization" in {
      implicit val f = binaryFormat[TestCase]
      canHandle(TestCase(Vector((Seq(12d, 2d), ZonedDateTime.now))))
    }

    "format time" in {
      val t = ZonedDateTime.now
      canHandle(t)
    }

    "format readings" in {
      val reading = (Seq(12d, 2d), ZonedDateTime.now)
      canHandle(reading)
    }

    "format seq" in {
      implicit val f = Macros.handler[TestCase2]
      canHandle(TestCase2(Seq(true, false)))
    }

    "composing" in {
      case class TestADT(a: String, b: Int)

      implicit val f = (
        field[String]("a1") ~
        field[Int]("b")
      )(TestADT.apply, unlift(TestADT.unapply))

      canHandle(TestADT("aa", 4))
    }

  }

}
