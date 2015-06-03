package glaux

import glaux.statistics.impl.apache.ApacheImplementations

package object statistics {
  val factory: DistributionImplementations = ApacheImplementations
}
