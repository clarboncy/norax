/**
 * TypeORM Issue #11637 Fix
 * 
 * Problem: Many-to-many relationships broken when user-defined junction entity exists (0.3.26 regression)
 * Root cause: PR #11114 fixed duplicate metadata but broke m2m persistence — junction inserts 
 * use DEFAULT/NULL instead of FK values because the relation's join columns aren't linked to 
 * the user-defined junction entity's columns.
 * 
 * Fix location: src/metadata-builder/EntityMetadataBuilder.ts
 * In the many-to-many relation processing block, check if a user-defined junction entity 
 * already exists in entityMetadatas before creating auto-generated metadata.
 * 
 * The fix has two parts:
 * 1. EntityMetadataBuilder.ts — reuse existing junction metadata if user-defined entity exists
 * 2. DataSource.ts — prefer user-defined junction metadata in entityMetadatasMap
 */

// ============================================================================
// PART 1: EntityMetadataBuilder.ts
// ============================================================================
// 
// FIND this block (around line 700+ in the many-to-many processing):
//
//   entityMetadata.relations
//       .filter((relation) => relation.isManyToMany)
//       .forEach((relation) => {
//           const joinTable =
//               this.metadataArgsStorage.findJoinTable(
//                   relation.target,
//                   relation.propertyName,
//               )!
//           if (!joinTable) return
//           // here we create a junction entity metadata for a new junction table of many-to-many relation
//           const junctionEntityMetadata =
//               this.junctionEntityMetadataBuilder.build(
//                   relation,
//                   joinTable,
//               )
//           relation.registerForeignKeys(
//               ...junctionEntityMetadata.foreignKeys,
//           )
//           relation.registerJoinColumns(
//               junctionEntityMetadata.ownIndices[0].columns,
//               junctionEntityMetadata.ownIndices[1].columns,
//           )
//           relation.registerJunctionEntityMetadata(
//               junctionEntityMetadata,
//           )
//           this.computeEntityMetadataStep2(junctionEntityMetadata)
//           this.computeInverseProperties(
//               junctionEntityMetadata,
//               entityMetadatas,
//           )
//           entityMetadatas.push(junctionEntityMetadata)
//       })
//
// REPLACE WITH:
//
//   entityMetadata.relations
//       .filter((relation) => relation.isManyToMany)
//       .forEach((relation) => {
//           const joinTable =
//               this.metadataArgsStorage.findJoinTable(
//                   relation.target,
//                   relation.propertyName,
//               )!
//           if (!joinTable) return
//
//           // Check if a user-defined junction entity already exists for this table
//           const existingJunctionMetadata = entityMetadatas.find(
//               (metadata) =>
//                   metadata.target === joinTable.name ||
//                   metadata.givenTableName === joinTable.name,
//           )
//
//           let junctionEntityMetadata: EntityMetadata
//           if (existingJunctionMetadata) {
//               // Reuse the user-defined junction entity metadata
//               // This preserves proper FK column references for persistence
//               junctionEntityMetadata = existingJunctionMetadata
//
//               // Ensure the relation's join columns point to the user-defined entity's columns
//               const joinColumns = this.junctionEntityMetadataBuilder
//                   .collectReferencedColumns(joinTable)
//               const inverseJoinColumns = this.junctionEntityMetadataBuilder
//                   .collectInverseReferencedColumns(joinTable)
//
//               // Match existing metadata columns with the referenced columns
//               const ownColumns = junctionEntityMetadata.ownColumns
//               const matchedJoinColumns = ownColumns.filter((col) =>
//                   joinColumns.some((jc) => jc.referencedColumn &&
//                       jc.referencedColumn.propertyName === col.referencedColumn?.propertyName),
//               )
//               const matchedInverseJoinColumns = ownColumns.filter((col) =>
//                   inverseJoinColumns.some((jc) => jc.referencedColumn &&
//                       jc.referencedColumn.propertyName === col.referencedColumn?.propertyName),
//               )
//
//               relation.registerForeignKeys(
//                   ...junctionEntityMetadata.foreignKeys,
//               )
//               relation.registerJoinColumns(
//                   matchedJoinColumns.length > 0 ? matchedJoinColumns : junctionEntityMetadata.ownIndices[0]?.columns || [],
//                   matchedInverseJoinColumns.length > 0 ? matchedInverseJoinColumns : junctionEntityMetadata.ownIndices[1]?.columns || [],
//               )
//               relation.registerJunctionEntityMetadata(
//                   junctionEntityMetadata,
//               )
//           } else {
//               // No user-defined junction entity — create auto-generated one (original behavior)
//               junctionEntityMetadata =
//                   this.junctionEntityMetadataBuilder.build(
//                       relation,
//                       joinTable,
//                   )
//               relation.registerForeignKeys(
//                   ...junctionEntityMetadata.foreignKeys,
//               )
//               relation.registerJoinColumns(
//                   junctionEntityMetadata.ownIndices[0].columns,
//                   junctionEntityMetadata.ownIndices[1].columns,
//               )
//               relation.registerJunctionEntityMetadata(
//                   junctionEntityMetadata,
//               )
//               this.computeEntityMetadataStep2(junctionEntityMetadata)
//               this.computeInverseProperties(
//                   junctionEntityMetadata,
//                   entityMetadatas,
//               )
//               entityMetadatas.push(junctionEntityMetadata)
//           }
//       })
//
// NOTE: JunctionEntityMetadataBuilder.collectReferencedColumns and 
// collectInverseReferencedColumns were made public in PR #11114.
// If they're still protected, change them to public.

// ============================================================================
// PART 2: DataSource.ts (entityMetadatasMap population)
// ============================================================================
//
// FIND the block where entityMetadatasMap is populated (around line 736-760):
//
// This should be a two-phase approach:
// 1. Add all non-junction (user-defined) entities first
// 2. Add junction entities only if they don't conflict with existing entries
//
// REPLACE the simple map population with:
//
//   // Phase 1: Add user-defined entities first (non-junction)
//   for (const metadata of entityMetadatas) {
//       if (!metadata.junction) {
//           this.entityMetadatasMap.set(metadata.target, metadata)
//       }
//   }
//   // Phase 2: Add junction entities only if not already present
//   for (const metadata of entityMetadatas) {
//       if (metadata.junction) {
//           if (!this.entityMetadatasMap.has(metadata.target)) {
//               this.entityMetadatasMap.set(metadata.target, metadata)
//           }
//       }
//   }

// ============================================================================
// PART 3: Test file
// ============================================================================
// Create: test/github-issues/11637/issue-11637.ts
// (See accompanying test file)