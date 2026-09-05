/**
 * Regression test for TypeORM issue #11637
 * Many-to-many relationships broken when user-defined junction entity exists
 *
 * This test verifies that junction table rows receive proper FK values
 * when a custom junction entity is defined.
 *
 * Place at: test/github-issues/11637/issue-11637.ts
 */

import "reflect-metadata"
import { expect } from "chai"
import { DataSource } from "../../../src/data-source/DataSource"
import { EntitySchema } from "../../../src/entity-schema/EntitySchema"
import { closeTestingConnections, createTestingConnections, reloadTestingDatabases } from "../../utils/test-utils"

describe("github issues > #11637 many-to-many broken with user-defined junction entity", () => {
    let connections: DataSource[]
    const __dirname = "."

    before(async () => {
        connections = await createTestingConnections({
            entities: [__dirname + "/entity/*"],
            enabledDrivers: ["postgres", "mysql", "sqlite", "better-sqlite3"],
        })
    })
    beforeEach(() => reloadTestingDatabases(connections))
    after(() => closeTestingConnections(connections))

    it("should persist junction table rows with proper FK values", async () => {
        for (const connection of connections) {
            // Create specialties
            const specialtyRepo = connection.getRepository("Specialty")
            const dogs = specialtyRepo.create({ name: "dogs" })
            const cats = specialtyRepo.create({ name: "cats" })
            await specialtyRepo.save([dogs, cats])

            // Create vet with specialties (many-to-many via user-defined junction)
            const vetRepo = connection.getRepository("Vet")
            const vet = vetRepo.create({
                name: "Carlos Salazar",
                specialties: [dogs, cats],
            })
            await vetRepo.save(vet)

            // Verify junction table has proper FK values (not NULL/DEFAULT)
            const junctionRepo = connection.getRepository("VetSpecialty")
            const junctionRows = await junctionRepo.find()

            expect(junctionRows.length).to.equal(2)
            // Each row should have non-null vet_id and specialty_name
            for (const row of junctionRows) {
                expect(row.vetId).to.not.be.null
                expect(row.vetId).to.not.be.undefined
                expect(row.specialtyName).to.not.be.null
                expect(row.specialtyName).to.not.be.undefined
                expect(row.specialtyName).to.be.oneOf(["dogs", "cats"])
            }

            // Verify the relation loads correctly
            const loadedVet = await vetRepo.findOne({
                where: { id: vet.id },
                relations: ["specialties"],
            })
            expect(loadedVet!.specialties.length).to.equal(2)
            expect(loadedVet!.specialties.map(s => s.name)).to.include.members(["dogs", "cats"])
        }
    })

    it("should not create duplicate junction metadata", async () => {
        for (const connection of connections) {
            const vetRepo = connection.getRepository("Vet")
            const metadata = vetRepo.metadata

            // Should have exactly one many-to-many relation
            const m2mRelations = metadata.relations.filter(r => r.isManyToMany)
            expect(m2mRelations.length).to.equal(1)

            // Junction metadata should exist and be the user-defined one
            const junctionMetadata = m2mRelations[0].junctionEntityMetadata
            expect(junctionMetadata).to.not.be.undefined
            expect(junctionMetadata!.target).to.equal("VetSpecialty")
        }
    })
})